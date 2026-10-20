# Applications and ideas

Every idea for ignite-scribe so far, with its status: what ships, what is in progress, what is planned, what is
only an idea, and what was tried and rejected (with the reason). New ideas go here first.

**Status key:** ✅ shipped · 🔨 in progress · 📋 planned · 💡 idea · ⛔ rejected or parked

## What it has access to, and why that matters

ignite-scribe sees almost everything a person reads on their PC, already as structured text rather than pixels:
- **content:** emails, chats, code, terminals, documents, browser pages, notifications, dialogs, captions;
- **structure:** the app, the window, the control (button, text box, table cell, tab), its position, what has
  focus;
- **time:** every change as it happens (the watcher), and optionally the last 7 days (history), searchable by words
  or by meaning;
- **the gaps:** OCR and UI-element detection, on the NPU, read what UI Automation cannot describe.

No other single source an agent can use holds all of this: files, browser history and an inbox are each a slice of
it. So every feature here is weighed twice: what it lets an agent do, and what it exposes. The defaults stay
"least data": redaction before anything leaves the process, opt-in for anything stored or continuous, and reads
the user can see.

**Next to similar projects.** Windows Recall and screenpipe record screenshots and OCR them; OmniParser-style agents
turn screenshots into clickable elements. ignite-scribe differs in four ways:
- it never copies an excluded window's pixels, and never stores pixels at all;
- it reads the OS's own structure first;
- it emits change events instead of frames;
- it runs alongside the user's work, at about 1.3% of a core for always-on OCR on the NPU.

## Realistic uses today

1. **An agent that sees your desktop safely:** what is on screen, the text of a dialog, which terminal shows an
   error, with passwords, excluded apps and secrets removed first.
2. **"What was that error an hour ago?"** Search the last 7 days by words or meaning: an error that scrolled away, a
   build number from a closed dialog, the page you had open before lunch.
3. **Apps without accessibility data:** games, remote desktops, canvas and Electron apps, old Win32 tools. NPU
   vision finds the controls and OCR reads their text.
4. **Remembering what you saw, not just what you opened:** with always-on OCR, text in video calls, image PDFs and
   remote sessions becomes searchable too.
5. **Waiting instead of polling:** an agent asks to be woken when a build finishes, rather than taking a screenshot
   every few seconds (1 call and 112 tokens, against 9,000-27,000 tokens of polling).
6. **Checking an agent's own actions:** after a click from a computer-use harness (such as Hermes), what changed,
   or that nothing did.
7. **Accessibility and handover:** summarize the day's work from history, or give a screen reader text that UI
   Automation cannot reach.

## ✅ Shipped

| Feature | Since | Notes |
|---|---|---|
| Window list (Win32): titles, processes, positions, z-order, focus | 0.0.1 | DPI-aware, physical pixels |
| Redaction policy: excluded apps and titles, secret scrubbing | 0.0.1 | Password managers, sign-in, banking and private-browsing windows; API keys, tokens, JWTs, Luhn-valid card numbers |
| stdio MCP server | 0.0.1 | Newline-delimited JSON-RPC; every reply marked untrusted |
| UI Automation text | 0.0.2 | Bounded walks; password fields are never read |
| Text pattern (terminals, documents) | 0.0.3 | |
| Change watcher, `screen_changes` | 0.0.4 | Desktop Duplication's dirty rectangles (never pixels) plus window events |
| `find_ui_elements`: UI elements on the NPU | 0.0.5 | OmniParser V1 detector, bit-exact on the NPU; excluded areas never copied |
| `ocr_windows` | 0.0.6 | Masked capture; text re-redacted |
| Encrypted history: `search_screen_history`, `screen_at` | 0.0.7 | AES-256-GCM under a DPAPI-wrapped key, 7-day retention, secure delete, no delete tool for agents |
| Semantic history search | 0.0.8 | bge-small on the iGPU (DirectML); vectors encrypted like the text |
| `wait_for`: wait for text, its absence, a change or quiet | 0.0.9 | 1 call against 6-12 polling turns |
| Action checker | 0.0.9 | `wait_for(since=..., quiet_s=0.5)` after an action reports what it changed |
| ScribeRec: a text recognizer trained for the NPU | 0.0.9 | Release-03 (after 0.1.0): 0.22% character errors on a real terminal against PP-OCRv5's 0.48%; exact on the NPU |
| Line cache for OCR | 0.0.9 | About 89% of lines reused on scrolling content |
| Always-on OCR (opt-in, NPU only) | 0.0.9 | Yields to NPU measurements, battery and pause; 1.3% of a core in the server |
| Prompt-injection flags, pattern tier | 0.1.0 | Hidden characters, overrides (EN/DE/FR/ES), role and address patterns; 0 false positives on UI lines and benign emails |
| Tray icon: shows when the screen is being read | 0.1.0 | Watching, reading pixels, paused or idle; pause from the menu |
| Tray polish and live settings | after 0.1.0 | Icons with shapes (no colour key needed), every setting in the menu, applied live; the background service owns the icon |
| One-line installer, `ignite-scribe setup`, background service | 0.1.0 | No admin rights, no Git; a per-user logon entry |
| `get_selection`: the text the user highlighted (idea 2) | after 0.1.0 | Text pattern of the focused control; policy first, password fields never read, scrubbed and flagged |
| `find_element`: the element map for clicking (idea 1) | after 0.1.0 | Name, role and window search; a click point the OS confirms reaches the control's window (menus and the taskbar included); enabled, accelerator and access keys, AutomationId, actions |

## 🔨 In progress

- **Secure mode:** a prompt-injection classifier trained on screen text that runs on the iGPU only, never the CPU.
  - Release-05 is installed (2026-10-20): attacks planted in email are caught at 12% -> 63% with no false alarms;
    it meets 5 of 7 pre-registered targets. Short, direct attacks (deepset, Gandalf) are the gap
    ([INJECTION.md](INJECTION.md)).
  - Scoring is incremental, at 75.6 ms per batch live.
  - A download for installed copies is next.
- **ScribeRec on busy terminal UIs:** release-03 beats PP-OCRv5 on a static terminal, but on a live TUI full of
  box-drawing rules and spinner glyphs it still makes about 3x PP-OCRv5's errors ([OCR.md](OCR.md)). Next: rule
  and box rows as neighbours in the generator.
- **A model download for installed copies:** ScribeRec, the detector and secure mode's model live only in local
  folders today.

## 📋 Planned (in the design)

- **Hook:** a short screen summary added to each prompt, for clients that support hooks. It is simpler than tools,
  but sends screen content with every prompt, so it stays opt-in ([DESIGN.md](DESIGN.md)).
- **One capture process that owns the NPU** (ignite-xdna's broker pattern), with the MCP server and the hook as its
  clients.
- **`find_on_screen`:** locate text or an element on screen by description.
- **A pause hotkey.** Pausing from the tray already works.
- **A public "attack the redaction" harness:** make the privacy claims something anyone can test. Most of the
  tests exist (masked pixels never enter the process, password values are never requested).
- **Battery cost of always-on OCR, in watts:** turn "1.3% of a core" into "X% of a battery day".

## 💡 Ideas, not started

### Helping agents act

| # | Idea | What it gives |
|---|---|---|
| 1 | ✅ **Element map for clicking**: `find_element("Save button in Notepad")` -> rectangle, centre, enabled, shortcut | Shipped as `find_element(name, role, match)`, see above. A click tool (e.g. Hermes's) hits exact coordinates instead of guessing from screenshots; `wait_for` then confirms. |
| 2 | ✅ **`get_selection`**: the text the user highlighted (UI Automation Text pattern) | Shipped, see above. "Explain this" or "fix this" without copy-paste |
| 3 | **Tables**: data grids as rows and columns (Excel, file lists, admin consoles, web tables) | Spreadsheets and dashboards become queryable |
| 4 | **App adapters**: terminal (folder, last command, exit status), IDE (file, line, Problems panel), browser (URL, tabs), mail and chat (sender, subject, messages) | Fields instead of raw text |
| 5 | **`expect` assertions**: `expect(window, text, within)` -> pass or fail | A test oracle for agents and UI scripts, built on `wait_for` |

### Push instead of poll

| # | Idea | What it gives |
|---|---|---|
| 6 | **Subscriptions**: triggers ("a terminal shows an error", "a new mail sender", "the build ends") pushed as MCP notifications | An agent waits at zero tokens until something happens |
| 7 | **MCP resources and prompts**: `screen://focused`, `screen://window/{id}`, `history://today`; prompts such as "summarize my day" | Works in any MCP client, no tool calls needed |

### Memory over time

| # | Idea | What it gives |
|---|---|---|
| 8 | **Day timeline**: time per app and project, from focus events | Standup notes, time sheets, "what did I do Tuesday?" |
| 9 | **Resume context (working memory)**: rebuild the windows, files, errors and pages of the last session on a project, with timestamped citations | "Pick up where I left off", answered from records, not guesses |
| 10 | **Error harvest**: every error and stack trace seen in a terminal or IDE, with when it first appeared | A debugging agent knows which error you looked at an hour ago |
| 11 | **Snippet provenance**: text that moved from one window to another | "Where did this snippet come from?" |
| 12 | **Missed notifications**: Windows toasts are readable | "What did I miss during the meeting?" |
| 13 | **Call captions**: Windows Live Captions is a readable window | A meeting transcript without touching audio; strictly opt-in |

### Safety

| # | Idea | What it gives |
|---|---|---|
| 14 | **Agent access log**: what each agent read, and when, in the tray and `ignite-scribe audit` | Users can see what was read; nothing similar seen elsewhere |
| 15 | **Screen-share secret guard**: warn when a secret or a password manager appears while sharing or recording | Data-loss prevention from the patterns that already exist |
| 16 | **Phishing check**: a sign-in page whose URL does not match the brand in its title; a fake dialog over another app | Catches lookalike pages |
| 17 | **Agent flight recorder**: an agent's actions with what each changed on screen, for replay and review | Accountability for computer-use agents |
| 18 | **Per-app consent**: approve apps one at a time from a tray prompt | Easier than an exclude list, and least-data by default |
| 19 | **Regression oracle**: record a workflow's redacted UI state, then diff a later build against it | Steadier and cheaper than screenshot comparison |

### Several agents

| # | Idea | What it gives |
|---|---|---|
| 20 | **Shared screen context**: several agents share one watcher and see what the others' actions did, with a lock on who drives the mouse | Agent teams on one desktop |

## ⛔ Rejected or parked, and why

| Idea | Why |
|---|---|
| An off-the-shelf injection classifier (protectai deberta-v3-base v2) | It flagged 62% of ordinary emails and 12% of UI lines ([INJECTION.md](INJECTION.md)); replaced by a classifier trained on screen text |
| Running the injection classifier on the NPU | No encoder path on the engine today; a port is engine work. It runs on the iGPU instead |
| Always-on OCR on the CPU | A standing CPU load is not acceptable; it runs on the NPU or not at all (the CPU path is for measurement) |
| Always-on OCR in the background service | Its text must join the live state of the agent's own server, so it stays there |
| PP-OCRv5 on the NPU through bf16 engine work | ScribeRec, a recognizer trained to fit the NPU's int8 engine, was chosen instead |
| The detector built with `--silu-sigmoid` | It cost 0.12-0.22 of icon recall here; the default build is bit-exact to the CPU fallback |
| Storing screenshots or OCR'd pixels | The privacy model: only redacted text and structure are ever stored |
