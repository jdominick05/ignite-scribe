# ignite-scribe

The desktop screen, turned into state any agent can read: which windows are open, what they say, and what you can
click. It is served over MCP (and, later, a hook). The OS's own structure comes first: Win32 windows and UI
Automation. NPU vision on XDNA1 (via [ignite-xdna](https://github.com/jdominick05/ignite-xdna)) handles only the pixels
the OS cannot describe.

**Status: 0.0.7.** Two sources: Win32 windows (no dependencies) and UI Automation (the `uia` extra: control
names, values, and the on-screen text of terminals and documents), kept current by a change-driven watcher
that re-reads only the windows that changed, with redaction and a stdio MCP server. For windows UI
Automation cannot describe, `find_ui_elements` finds UI elements (boxes, no text) with an icon detector
running on the XDNA1 NPU (see [docs/VISION.md](docs/VISION.md)), and `ocr_windows` reads their text with
PP-OCRv5, on the CPU until an NPU-native recognizer is trained ([docs/OCR.md](docs/OCR.md)). An opt-in, encrypted
screen history answers "what was on screen when". See [docs/DESIGN.md](docs/DESIGN.md) for the plan.

```powershell
python -m ignite_scribe snapshot          # the redacted screen state once, as markdown (--json, --no-text)
python -m ignite_scribe watch --seconds 30 # change events as JSON lines (Ctrl+C to stop)
python -m ignite_scribe mcp               # MCP tools on stdio: scribe_status, get_screen_state,
                                          #   screen_changes, find_ui_elements, ocr_windows,
                                          #   search_screen_history, screen_at, list_windows
python -m ignite_scribe history status    # the opt-in history: status, search QUERY --since 2h, purge
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
Duplication's changed-rectangle list and from window events, without reading pixels, and it holds its change
events in memory only, unless history is on.

**History is opt-in, encrypted, and kept 7 days.** Off unless `IGNITE_SCRIBE_HISTORY=1` or the policy file sets
`"history": true`; then the MCP server records from its start until its client exits.
- It records only what the watcher already redacted: change events and, while the screen changes, a snapshot every
  5 minutes. Excluded windows appear without title or content, and secrets are already scrubbed. No pixels, and no
  OCR text.
- The file is `%LOCALAPPDATA%\ignite-scribe\history.db` (`IGNITE_SCRIBE_HISTORY_DB` overrides it). Each chunk is
  sealed with AES-256-GCM (Windows CNG) under a random key that is stored only wrapped with DPAPI for your Windows
  account: another account, or the file copied to another machine, cannot read it.
- Chunks older than 7 days are deleted, and the oldest go first above 200 MB.
- `search_screen_history` (words, a window filter, a time range) and `screen_at` (the screen at a past time) read
  it. Agents cannot delete it; `python -m ignite_scribe history purge` deletes it all and replaces the key.

**Vision reads pixels, and only when called.**
- `find_ui_elements` and `ocr_windows` copy the on-screen pixels of the windows they look at, in memory, never
  to disk.
- It never copies windows above them, excluded windows, or password fields. Those areas are left grey.
- It skips elevated windows unless the policy sets `capture_elevated`.
- OCR text is redacted like UI Automation text, and lines lying under masked areas are dropped.
- The vision worker starts on the first call and exits after 30 s idle, releasing the NPU.
- `IGNITE_SCRIBE_VISION=0` turns both tools off.

The detector is OmniParser V1's `icon_detect`, under AGPL-3.0; it is not shipped here. Building it for the
NPU needs [ignite-xdna](https://github.com/jdominick05/ignite-xdna); docs/VISION.md has the steps.

Tests: `python -m pytest -q` (offline, plus live Win32 and UI Automation tests on Windows).

Licence: AGPL-3.0-or-later.
