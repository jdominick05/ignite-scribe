# ignite-scribe

The desktop screen, turned into state any agent can read: which windows are open, what they say, and what you can
click. It is served over MCP (and, later, a hook). The OS's own structure comes first: Win32 windows and UI
Automation. NPU vision on XDNA1 (via [ignite-xdna](https://github.com/jdominick05/ignite-xdna)) handles only the pixels
the OS cannot describe.

**Status: 0.0.4.** Two sources: Win32 windows (no dependencies) and UI Automation (the `uia` extra: control
names, values, and the on-screen text of terminals and documents), kept current by a change-driven watcher
that re-reads only the windows that changed, with redaction and a stdio MCP server. No OCR yet. See [docs/DESIGN.md](docs/DESIGN.md) for the plan.

```powershell
python -m ignite_scribe snapshot          # the redacted screen state once, as markdown (--json, --no-text)
python -m ignite_scribe watch --seconds 30 # change events as JSON lines (Ctrl+C to stop)
python -m ignite_scribe mcp               # MCP tools on stdio: scribe_status, get_screen_state,
                                          #   screen_changes, list_windows
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
- password fields are reported as `[password field]`; their value and text are never requested;
- terminals and documents are read as far as they are visible on screen (not their scrollback), so a terminal's
  output reaches the agent: exclude a process in the policy file to keep it out;
- secret-shaped text (API keys, tokens, private keys, JWTs, Luhn-valid card numbers) is replaced;
- `IGNITE_SCRIBE_POLICY` names a JSON policy file that extends the defaults.

Every result is also marked as untrusted, because text on screen can be written to instruct an agent.

**While it watches.** The MCP server starts its watcher on the first screen tool call and keeps it until the
client exits; `IGNITE_SCRIBE_WATCH=0` turns it off. It learns *where* the screen changed from Desktop
Duplication's changed-rectangle list and from window events. It never reads the pixels, and it holds its
change history in memory only.

Tests: `python -m pytest -q` (offline, plus live Win32 and UI Automation tests on Windows).

Licence: AGPL-3.0-or-later.
