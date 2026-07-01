"""
test_no_subprocess_bypass.py
Vector 2 (P0) gate — adaptive_hunt_orchestrator.py must contain zero direct
subprocess.run() calls whose argument list targets worker_cupy.py or
validation_pipeline.py.
After the IPC routing patch, all such calls are routed through
_dispatch_worker / _dispatch_validator helpers.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "adaptive_hunt_orchestrator.py"

_DIRECT_SCRIPTS = {"worker_cupy.py", "validation_pipeline.py"}

# Functions that are themselves the dispatch abstraction layer — subprocess
# references inside them are the intentional single fallback point.
_EXEMPT_FUNCTIONS = {"_dispatch_worker", "_dispatch_validator"}


class _SubprocessCallVisitor(ast.NodeVisitor):
    """Collect subprocess.run() cmd-arg strings from all non-exempt functions."""

    def __init__(self):
        self.violations: list[tuple[str, list[str]]] = []
        self._current_func: str = "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev = self._current_func
        self._current_func = node.name
        if node.name not in _EXEMPT_FUNCTIONS:
            self.generic_visit(node)
        self._current_func = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call):
        func = node.func
        is_subprocess_run = (
            isinstance(func, ast.Attribute)
            and func.attr == "run"
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        )
        if is_subprocess_run and node.args:
            first = node.args[0]
            strings: list[str] = []
            if isinstance(first, (ast.List, ast.Tuple)):
                for elt in first.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        strings.append(elt.value)
            if strings:
                self.violations.append((self._current_func, strings))
        self.generic_visit(node)


def test_no_direct_worker_cupy_subprocess():
    """Outside the dispatch helpers, no subprocess.run call may target worker_cupy.py."""
    source = TARGET.read_text(encoding="utf-8")
    visitor = _SubprocessCallVisitor()
    visitor.visit(ast.parse(source))
    violations = [
        (fn, args) for fn, args in visitor.violations
        if any("worker_cupy.py" in a for a in args)
    ]
    assert not violations, (
        f"Direct worker_cupy.py subprocess.run calls still present outside dispatch helpers "
        f"in adaptive_hunt_orchestrator.py: {violations}"
    )


def test_no_direct_validation_pipeline_subprocess():
    """Outside the dispatch helpers, no subprocess.run call may target validation_pipeline.py."""
    source = TARGET.read_text(encoding="utf-8")
    visitor = _SubprocessCallVisitor()
    visitor.visit(ast.parse(source))
    violations = [
        (fn, args) for fn, args in visitor.violations
        if any("validation_pipeline.py" in a for a in args)
    ]
    assert not violations, (
        f"Direct validation_pipeline.py subprocess.run calls still present outside dispatch helpers "
        f"in adaptive_hunt_orchestrator.py: {violations}"
    )


def test_dispatch_helpers_present():
    """Confirm _dispatch_worker and _dispatch_validator are defined in the module."""
    source = TARGET.read_text(encoding="utf-8")
    tree = ast.parse(source)
    defs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "_dispatch_worker" in defs, "_dispatch_worker helper not found"
    assert "_dispatch_validator" in defs, "_dispatch_validator helper not found"


def test_queue_conditional_import_present():
    """Confirm the _QUEUE_AVAILABLE guard and _QueueManager import exist."""
    source = TARGET.read_text(encoding="utf-8")
    assert "_QUEUE_AVAILABLE" in source, "_QUEUE_AVAILABLE flag missing from module"
    assert "_QueueManager" in source, "_QueueManager import missing from module"
