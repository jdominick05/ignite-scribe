# ignite-scribe

The desktop screen, turned into state any agent can read: which windows are open, what they say, and what you can
click. It is served over MCP (and, later, a hook). The OS's own structure comes first: Win32 windows and UI
Automation. NPU vision on XDNA1 (via [ignite-xdna](https://github.com/jdominick05/ignite-xdna)) handles only the pixels
the OS cannot describe.

**Status: 0.0.2.** Two sources: Win32 windows (no dependencies) and UI Automation (the `uia` extra), with
redaction and a stdio MCP server. No pixel capture or OCR yet. See [docs/DESIGN.md](docs/DESIGN.md) for the plan.

```powershell
python -m ignite_scribe snapshot          # the redacted screen state once, as markdown (--json, --no-text)
python -m ignite_scribe mcp               # MCP tools on stdio: scribe_status, get_screen_state, list_windows
```

Install it (`pip install -e .`, or `pip install -e .[uia]` for UI Automation), then register it with any MCP
client as a stdio server:

```json
{"mcpServers": {"ignite-scribe": {"command": "python", "args": ["-m", "ignite_scribe", "mcp"]}}}
```

**What it shares.** Everything an agent reads from this server may go to a remote model, so redaction runs before
anything leaves the engine:
- password-manager processes and sign-in / banking / private-browsing titles are withheld by default;
- an excluded window is never opened with UI Automation, so its content never enters the engine;
- password fields are reported as `[password field]`; their value is never requested;
- secret-shaped text (API keys, tokens, private keys, JWTs, Luhn-valid card numbers) is replaced;
- `IGNITE_SCRIBE_POLICY` names a JSON policy file that extends the defaults.

Every result is also marked as untrusted, because text on screen can be written to instruct an agent.

Tests: `python -m pytest -q` (offline, plus live Win32 and UI Automation tests on Windows).

Licence: AGPL-3.0-or-later.
