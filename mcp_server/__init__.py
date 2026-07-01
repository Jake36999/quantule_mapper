"""
mcp_server — read-only MCP interface for the IRER / Quantule Mapper stack.

NOTE: the package is named `mcp_server` (not `mcp` as sketched in
docs/MCP_TOOLS_SPEC.md §2) because a top-level `mcp` package would shadow the
`mcp` SDK (`from mcp.server.fastmcp import FastMCP`) on import.

Layering:
  * config.py       — path/DB resolution + read whitelist (no SDK, no GPU)
  * data_access.py  — pure read-only query layer (no SDK, no GPU) -> fully unit-testable
  * server.py       — thin FastMCP entrypoint registering the tools

This MVP implements the 8 read-only tools from MCP_TOOLS_SPEC.md §6.  Write/GPU
tools (run_smoke_simulation, stage/run_simulation_manifest, validate_artifact)
are intentionally not implemented here.
"""
