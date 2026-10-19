# Design

## The idea
An agent working on a desktop needs to know what is on it. Screenshots are expensive to send and to read, and they
leak everything on screen. ignite-scribe keeps a structured, redacted model of the screen up to date and serves
it on request. The OS can describe most of a desktop exactly and for free; vision runs only on what it cannot.

## Sources, in order
1. **Win32 windows** (done): visible top-level windows, z-order, focus, bounds, owning process; monitors.
2. **UI Automation** (next): control trees of the focused and visible windows (names, roles, values, bounds,
   document text). Password fields (`IsPassword`) are never read.
3. **Change-driven capture**: Desktop Duplication reports the dirty rectangles per frame, so only changed regions
   are reprocessed. A static desktop costs nearly nothing.
4. **NPU vision** for regions UI Automation leaves blank (images, video, games, remote desktops, canvas apps),
   through ignite-xdna:
   - UI element detection: OmniParser's detector is a fine-tuned YOLOv8 (AGPL), the engine's strongest
     measured case;
   - text detection and recognition (OCR). Windows' built-in OCR is the CPU fallback and the baseline to beat.
5. **History**: snapshots in SQLite with a retention limit, plus embeddings for "what was on screen when" (batched,
   since short single embeddings lose to the iGPU on this NPU).

## Consumers
- **MCP server** (done, stdio): the agent asks only when it needs the screen. Tools: `scribe_status`,
  `get_screen_state`, `list_windows`. Planned: `find_on_screen`, `screen_changes(since)`,
  `search_screen_history`.
- **Hook** (planned): a short screen summary on each prompt. It is simpler, but it sends screen content with every
  request, so it stays opt-in.

The capture engine will run as one long-lived process that owns the NPU (the ignite-xdna broker pattern). The MCP
server and the hook will be its clients.

## Privacy model
- Local only: nothing is sent anywhere by the engine itself. Consumers decide what leaves the machine.
- Redaction at the source, before any consumer, history included. It covers excluded processes and title patterns
  (the window keeps its position, and its title and content are withheld), plus secret-shaped text.
- Every output carries the untrusted-content note. Screen text is data, never instructions.
- Planned: a pause hotkey, a running indicator, and no raw screenshots kept by default.

## Prior art (checked 2026-10-19)
- screenpipe: continuous capture, accessibility first, OCR fallback, SQLite + MCP. CPU, about 1 frame a second.
- Windows MCP servers (windows-gui-mcp, mcp-test-utils, win32-mcp-server): UI Automation / OCR on demand, CPU.
- OmniParser V2: vision-only UI parsing, YOLOv8 + Florence-2, sized for big GPUs.
- Windows Recall: needs a 40+ TOPS NPU, so XDNA1 is excluded.

None of them runs on an XDNA1 NPU, or keeps a change-driven structured state.
