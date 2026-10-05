"""Harness registry + component-hash provenance (IMPLEMENTATION_PLAN_2026-10 Phase D)."""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, ROOT)

import build_harness_registry as reg  # noqa: E402
import stepper_staleness as stal  # noqa: E402
from jax_scout import provenance  # noqa: E402

import pytest  # noqa: E402


def _git_ok():
    """A git worktree created on Windows stores a Windows path in .git, which WSL's git cannot follow.
    History-dependent tests skip there rather than report a false failure; CI (a plain clone) runs them."""
    try:
        r = subprocess.run(["git", "rev-parse", "e270cdc"], cwd=ROOT, capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


needs_git = pytest.mark.skipif(not _git_ok(), reason="git history not readable from this environment")

GOOD = '''"""doc"""
HARNESS = {"id": "x", "branch": "Branch - Gravity - Index", "status": "ACTIVE", "superseded_by": None,
           "invariants": [], "produces": ["X_RUN"], "summary": "what it answers"}
import os
OUT = os.path.join("sweep_runs", "X_RUN")
'''


def test_manifest_is_read_without_importing(tmp_path):
    src = GOOD + "\nimport this_module_does_not_exist\n"      # would fail on import
    man, err = reg.read_manifest(src)
    assert err is None and man["produces"] == ["X_RUN"] and reg.validate(man) == []


def test_non_literal_manifest_is_an_error_not_an_exception():
    man, err = reg.read_manifest('HARNESS = dict(id=compute())\n')
    assert man is None and "literal" in err


def test_validation_rules():
    assert reg.validate({"status": "BOGUS", "branch": "b", "summary": "s"})
    assert reg.validate({"status": "SUPERSEDED", "branch": "b", "summary": "s"})   # needs superseded_by
    assert not reg.validate({"status": "SUPERSEDED", "superseded_by": "y", "branch": "b", "summary": "s"})


def test_check_flags_a_sweep_writer_without_manifest(tmp_path, monkeypatch):
    d = tmp_path / "jax_scout"
    d.mkdir()
    (d / "bare.py").write_text('import os\nos.makedirs("sweep_runs/BARE_RUN")\n', encoding="utf-8")
    (d / "doc_only.py").write_text('"""writes to sweep_runs/ (docstring only)"""\nimport os\nos.makedirs("x")\n',
                                   encoding="utf-8")
    (d / "ok.py").write_text(GOOD, encoding="utf-8")
    (d / "lib.py").write_text("def f(): return 1\n", encoding="utf-8")       # never writes runs
    monkeypatch.setattr(reg, "ROOT", str(tmp_path))
    assert reg.check([str(d / "ok.py"), str(d / "lib.py"), str(d / "doc_only.py")]) == 0
    assert reg.check([str(d / "bare.py")]) == 1


def test_every_declared_manifest_in_the_repo_is_valid():
    bad = [e for e in reg.scan() if e["manifest"] and e["error"]]
    assert not bad, bad


def test_every_manifested_script_still_compiles():
    """ast.parse accepts a statement placed above `from __future__ import ...`; compile() does not.
    The first manifest insertion (2026-10-04) made exactly that mistake in three harnesses."""
    for e in reg.scan():
        if e["manifest"]:
            src = open(os.path.join(ROOT, e["file"]), encoding="utf-8").read()
            compile(src, e["file"], "exec")


@needs_git
def test_blob_hash_matches_git_hash_object():
    for rel in ("jax_scout/provenance.py", "solver/etdrk4_coeffs.py", "tools/stepper_staleness.py"):
        want = subprocess.run(["git", "hash-object", rel], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
        assert provenance.git_blob_hash(os.path.join(ROOT, rel)) == want, rel


@needs_git
def test_component_hashes_cover_imported_repo_modules_and_flag_dirty(tmp_path):
    import solver.etdrk4_coeffs  # noqa: F401
    hashes, dirty = provenance.component_hashes()
    assert "solver/etdrk4_coeffs.py" in hashes and "jax_scout/provenance.py" in hashes
    assert not any(".venv" in k or "site-packages" in k for k in hashes)
    tree = provenance._head_tree()
    for rel in hashes:
        assert (rel in dirty) == (tree.get(rel) != hashes[rel])


@needs_git
def test_hash_verdict_uses_the_fixed_files_content():
    fix = {"fix_commit": "e270cdc", "affects_files": ["solver/kernels.py"]}
    post = subprocess.run(["git", "rev-parse", "e270cdc:solver/kernels.py"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    pre = subprocess.run(["git", "rev-parse", "dda7bd2:solver/kernels.py"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    assert stal.hash_verdict({"solver/kernels.py": post}, fix) is True
    assert stal.hash_verdict({"solver/kernels.py": pre}, fix) is False
    assert stal.hash_verdict({"other.py": pre}, fix) is None            # no evidence -> fall back
    assert stal.hash_verdict({"solver/kernels.py": "0" * 40}, fix) is False   # unknown blob -> stale


@needs_git
def test_hash_evidence_overrides_a_misleading_commit():
    """A run stamped with a post-fix commit but a dirty, pre-fix kernels.py must come out STALE."""
    pre = subprocess.run(["git", "rev-parse", "dda7bd2:solver/kernels.py"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    fix = {"id": "f", "fix_commit": "e270cdc", "fix_date": "2026-10-02",
           "affects_steppers": ["ETDRK4-cupy"], "affects_files": ["solver/kernels.py"]}
    st = stal.staleness("R", ["ETDRK4-cupy"], "e270cdc", "2026-10-05", fix,
                        component_hashes={"solver/kernels.py": pre})
    assert st[0] == "STALE_PENDING_REVALIDATION" and "hash" in st[1]
