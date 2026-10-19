# Design

## The idea
An agent working on a desktop needs to know what is on it. Screenshots are expensive to send and to read, and they
leak everything on screen. ignite-scribe keeps a structured, redacted model of the screen up to date and serves
it on request. The OS can describe most of a desktop exactly and for free; vision runs only on what it cannot.

## Sources, in order
1. **Win32 windows** (done): visible top-level windows, z-order, focus, bounds, owning process; monitors.
2. **UI Automation** (done): the control trees of the focused window, then the topmost ones (names, roles,
   values, bounds). Password fields (`IsPassword`) are never read. Each walk is bounded (node cap, deadline, UI
   Automation's own timeouts). A whole pass over 4 windows on Desktop 2, one sitting: 136-139 ms warm, 234 ms
   cold, with Windows Terminal's visible text (34 lines) included; about 120 ms before the Text pattern.
   Known gaps:
   - elevated apps (Task Manager) are hidden from a normal process by UIPI, and are noted as unreadable;
   - the Text pattern (since 0.0.3) is read for its visible ranges only, so scrollback and off-screen pages are
     not in the snapshot. Its multi-line text replaces the element's subtree, so a document's links are not
     listed separately.
3. **Change-driven capture** (done, `watch.py`): one background thread.
   - **Change signals:**
     - WinEvents: focus, create / destroy / show / hide / move / minimize, and name or value changes inside a
       window;
     - Desktop Duplication (`dirty.py`): the changed and moved rectangles per frame, mapped to the topmost
       windows under them. Only the rectangle metadata is read; the desktop image is never mapped.
   - **Refreshes:** at most every 0.25 s, re-read the window list and walk only the dirty windows among the
     ones worth reading; the rest keep their content.
   - **Events:** diff the redacted state (`changes.py`) into numbered events in an in-memory ring.
     `screen_changes` merges repeats per window by default.
   - **Fallback:** without duplication (secure desktop, an exclusive full-screen app), the focused window is
     re-walked every 2 s.
   - **Measured** on Desktop 2, one sitting, with this terminal animating its title and status lines (not an
     idle desktop):
     - 15-48 refreshes in 4-15 s windows, 26-36 ms each on average;
     - the watcher thread used 4.3-7.4% of one core;
     - `get_screen_state` from its state returns in under 1 ms, against about 120-140 ms for a fresh walk.
     - A truly idle desktop was not measured.
4. **NPU vision** (UI elements done, `find_ui_elements`; see VISION.md) for regions UI Automation leaves
   blank (images, video, games, remote desktops, canvas apps), through ignite-xdna:
   - UI element detection: OmniParser's detector is a fine-tuned YOLOv8 (AGPL), the engine's strongest
     measured case;
   - text detection and recognition (OCR): done on the CPU, `ocr_windows` (PP-OCRv5 mobile, four times fewer
     character errors than Windows' built-in OCR). The recognizer moves to the NPU when ignite-xdna's bf16
     datapath can run it (see OCR.md).
5. **History**: snapshots in SQLite with a retention limit, plus embeddings for "what was on screen when" (batched,
   since short single embeddings lose to the iGPU on this NPU).

## Consumers
- **MCP server** (done, stdio): the agent asks only when it needs the screen. Tools:
  - `scribe_status` (with the watcher's state);
  - `get_screen_state`: served from the watcher's state; `refresh` walks afresh;
  - `screen_changes(since)`: events since a sequence number, with `next_since`;
  - `list_windows`: no content. Planned: `find_on_screen`, `search_screen_history`.
- **Hook** (planned): a short screen summary on each prompt. It is simpler, but it sends screen content with every
  request, so it stays opt-in.

The capture engine will run as one long-lived process that owns the NPU (the ignite-xdna broker pattern). The MCP
server and the hook will be its clients.

## Privacy model
- Local only: nothing is sent anywhere by the engine itself. Consumers decide what leaves the machine.
- Redaction at the source, before any consumer, history included. It covers excluded processes and title patterns
  (the window keeps its position, and its title and content are withheld), plus secret-shaped text.
- Every output carries the untrusted-content note. Screen text is data, never instructions.
- Pixels are read only by the vision stage, only per call, and never written anywhere.
  - Exclusion is applied before capture: masked areas are never blitted. They cover windows above the target,
    excluded windows, and password fields.
  - Elevated windows are skipped unless `capture_elevated` is set.
- Planned: a pause hotkey and a running indicator.

## Prior art (checked 2026-10-19)
- screenpipe: continuous capture, accessibility first, OCR fallback, SQLite + MCP. CPU, about 1 frame a second.
- Windows MCP servers (windows-gui-mcp, mcp-test-utils, win32-mcp-server): UI Automation / OCR on demand, CPU.
- OmniParser V2: vision-only UI parsing, YOLOv8 + Florence-2, sized for big GPUs.
- Windows Recall: needs a 40+ TOPS NPU, so XDNA1 is excluded.

None of them runs on an XDNA1 NPU, or keeps a change-driven structured state.
