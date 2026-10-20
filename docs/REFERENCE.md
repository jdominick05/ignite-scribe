# Reference

Everything ignite-scribe does with your screen, in detail: what it withholds, what it stores, and what each
optional part reads. The [README](../README.md) is the short version. The design and its reasons are in
[DESIGN.md](DESIGN.md).

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

**Injection flags.** Text that looks written to instruct an agent gets a `flags` entry on its window (and on the
change event that brought it), with the kind, an excerpt, where it came from (title, UI Automation or OCR) and the
app category (browser, mail, chat, terminal, editor). A flag is metadata, never a filter, and a window without one is
not thereby safe.
- It runs locally at capture time, on already-scrubbed text, so an excerpt never holds a secret. Withheld windows
  are never scanned.
- Before matching, the text is normalized (NFKC, casefold, invisible characters dropped).
- Invisible characters are a finding of their own: Unicode tag characters, bidi controls, zero-width runs. The
  user cannot see them, but a model reads them.
- This first tier is patterns: English, plus German, French and Spanish overrides. It is cheap, adding 0.23 ms
  to a watcher refresh with its cache, and precise, but it misses most injections that are phrased as ordinary
  requests:

| set (`tools/injection_eval.py`, test splits) | recall | false positives |
|---|---:|---:|
| deepset/prompt-injections (direct, EN/DE) | 26.7% of 60 | 0 of 56 |
| Lakera/gandalf_ignore_instructions (direct) | 64.3% of 112 | n/a |
| microsoft/BIPIA, attacks planted in emails (indirect) | 12.0% of 250 | 0 of 50 emails |
| synthetic UI lines (seed 100) | n/a | 0 of 500 |
| this desktop, 60 s of the watcher's state | n/a | 0 flagged windows |

An off-the-shelf classifier was measured and rejected: it flags 62% of ordinary emails and 12% of UI lines
([INJECTION.md](INJECTION.md)). A classifier trained on screen text is in progress. The patterns were tuned on the
train splits only.

**Waiting instead of polling.** `wait_for` blocks until text appears (`text`), goes (`gone`), or anything
changes, and with `quiet_s` until the screen settles. Text already on screen counts, and so does text that came and
went after `since`, so nothing between two calls is missed. An agent that clicks or types with its own tools (a
computer-use harness) can check the action: note `next_since`, act, then call `wait_for(since=..., quiet_s=0.5)`.
The reply lists what the action changed, and `timeout` means it changed nothing.

**Where to click.** `find_element(name, role, match)` walks the matched windows fresh (the policy first, so an
excluded window is never read) and returns controls ranked by name: exact, prefix, a word's prefix, substring,
then AutomationId and value. The focused window comes first on ties. Each hit has:
- `rect` and `click`, in physical screen pixels (per-monitor DPI aware). A click tool that is not DPI aware must
  divide by the display scale;
- `click` is a point the OS confirmed (`click_checked`): `WindowFromPoint` there, then its root, is the hit's
  window. So menus, tooltips, the taskbar and topmost tool windows, which the window list leaves out, are
  accounted for. The centre is tried first, then the visible pieces' centres; null means nothing reaches it;
- `visible_share`, an estimate from rectangles only: the control clipped to its window and the monitors, minus
  the listed windows above it (off-screen parts count as not visible);
- `enabled`, `accelerator` (e.g. Ctrl+S, while the app has focus), `access_key` (e.g. `Alt+s` from a `&Save`
  label, only while that dialog or menu is active), `automation_id`, `focused` and `actions` (invoke, toggle,
  select, expand);
- its window (hwnd, title, process, z) and whether that window's text is flagged.

A password field is found with its rectangle and role only. Names are screen text, so a page can label anything
"Save": check the hit's window. Then `wait_for(since=seq, quiet_s=0.5)` confirms what the click did.
Snapshots, the watcher and history do not carry this detail, so their content and events are unchanged.
Cost (Desktop 2, 5 calls each): the focused window alone, median 27 ms; four windows, median 842 ms (3 had
content, 122 controls, 118 of them partly covered by a window above).

What noticing the end of a 20 s build costs (`tools/waitfor_bench.py`, Desktop 2, 3 runs; text tokens estimated as
bytes/4; screenshot tokens as w*h/750 after scaling to at most 1568 px a side and 1.15 MP, a common vision-model
rule):

| | model turns | tokens to the model | noticed after |
|---|---:|---:|---:|
| `wait_for` | 1 | 112 | 56-247 ms |
| text polling every 5 s (`get_screen_state`) | 6 | about 27,300 | 2.5 s on average |
| screenshot polling every 5 s | 6 | about 9,200 | 2.5 s on average |
| screenshot polling every 2 s | 12 | about 18,400 | 1 s on average |

Each polling turn also re-reads the agent's context, which this table does not count.

**The tray icon.** While any MCP session runs, one dot in the notification area shows what ignite-scribe is
doing. Each state has its own shape as well as its own colour:
- a blue eye: watching windows and their UI Automation text, no pixels;
- amber lines of text: reading pixels (`find_ui_elements`, `ocr_windows`, always-on OCR);
- white pause bars: always-on OCR is paused (a `find_ui_elements` or `ocr_windows` call still shows as reading);
- a grey slashed ring: not reading the screen.

A red dot at its lower right means history is being recorded. Hovering shows the same in words, with history and
always-on OCR on their own lines. Clicking opens a menu with that status, how many ignite-scribe processes are
running, a key to the icons, and "Pause always-on OCR" when it is running (otherwise it says always-on OCR is off).
The background service draws the icon whenever it runs; agents' servers draw it only when there is no service.

**Settings from the tray.** The menu's Settings submenu changes what `ignite-scribe setup` sets, in the same
policy file: history, always-on OCR, secure mode, whether vision may read administrator windows, and running in
the background from logon. Under "Apps never read" you can exclude any app with an open window, or include one
again; the built-in exclusions are listed and cannot be removed. Window-title patterns and anything else are
edited in the policy file itself ("Edit the policy file..."). Always-on OCR and secure mode can be chosen before
their models are installed, as in setup; the menu says so, and they start once the model is there.

Every running ignite-scribe process applies a change within about two seconds, without a restart. Turning
history off closes the store in every process, so none goes on recording or takes recording over. Turning
always-on OCR or secure mode off stops it. A new exclusion applies to what the watcher shows at once. A process
from an older version cannot reload its settings; the menu names it, so you can restart its agent session.

Each session writes a small status file with flags only, no screen content, under
`%LOCALAPPDATA%\ignite-scribe\status`. `IGNITE_SCRIBE_TRAY=0` turns the icon off.

**While it watches.** The MCP server starts its watcher on the first screen tool call and keeps it until the
client exits; `IGNITE_SCRIBE_WATCH=0` turns it off. It learns *where* the screen changed from Desktop
Duplication's changed-rectangle list and from window events, without reading pixels, and it holds its change
events in memory only, unless history is on.

**History is opt-in, encrypted, and kept 7 days.** Off unless `IGNITE_SCRIBE_HISTORY=1` or the policy file sets
`"history": true`; then the MCP server records from its start until its client exits.
- It records only what the watcher already redacted: change events and, while the screen changes, a snapshot every
  5 minutes. Excluded windows appear without title or content, and secrets are already scrubbed. No pixels; OCR
  text only from always-on OCR, when that is on too and runs in the recording process (per-call `ocr_windows`
  results are never recorded). The background service records but does not run always-on OCR, so with the
  service on, always-on text reaches agents live but is not recorded.
- The file is `%LOCALAPPDATA%\ignite-scribe\history.db` (`IGNITE_SCRIBE_HISTORY_DB` overrides it). Each chunk is
  sealed with AES-256-GCM (Windows CNG) under a random key that is stored only wrapped with DPAPI for your Windows
  account: another account, or the file copied to another machine, cannot read it.
- Chunks older than 7 days are deleted (overwritten, not left in free pages), and the oldest go first above
  200 MB. This runs on open and every 10 minutes, and searches never reach past 7 days.
- With several client sessions open, one server records and the others only search; when it exits, another
  takes over.
- `search_screen_history` (words, a window filter, a time range) and `screen_at` (the screen at a past time) read
  it. Agents cannot delete it; `python -m ignite_scribe history purge` deletes it all and replaces the key.
- **Semantic search** (`mode: "semantic"`) finds passages by meaning ("the article about GPU prices"). Each
  window's text over 5 minutes becomes passages of about 200 tokens, which the recording server embeds in the
  background with bge-small-en-v1.5 (MIT). Passages are embedded on the iGPU through DirectML, in a worker process
  at below-normal priority that exits after 60 s idle or when the server goes; queries are embedded on the CPU.
  The vectors are encrypted like the text, since text can be partly recovered from an embedding.
  `IGNITE_SCRIBE_EMBED=0` turns it off. Setup: `pip install -e .[embed]`, put `model.onnx` and `tokenizer.json`
  from `BAAI/bge-small-en-v1.5` (its `onnx/` folder) in `models/embed/bge-small-en-v1.5/`, then
  `python tools/make_embed_fp16.py` (needs `onnx onnxconverter-common`) for the faster fp16 copy.

**Secure mode (opt-in, iGPU only; its model is in training).** `"secure_mode": true` in the policy, or
`IGNITE_SCRIBE_SECURE=1`. A classifier fine-tuned on screen text (INJECTION.md, tier 2) scores each window's
redacted text.
- It runs in the embedding worker on DirectML, at below-normal priority, and exits after 60 s idle. When DirectML
  is unavailable it stays off; it never runs on the CPU. `scribe_status` says why.
- Scoring is asynchronous and incremental: a window's text is cut into pieces of about 1,500 characters, each
  scored once; a new piece is queued, and the window's `classifier` flag (with its score) appears on the next
  refresh. Scores are kept by a digest of the text, not the text.
- Withheld windows are never scored.
- The model is looked up in `IGNITE_SCRIBE_SECURE_MODEL`, then `%LOCALAPPDATA%\ignite-scribe\models\injection\scribe-guard`.

**Always-on OCR (opt-in, NPU only).** `IGNITE_SCRIBE_ALWAYS_ON=1` or
`"always_on_ocr": true` in the policy. One server per machine reads, as their pixels change, the text of windows UI
Automation cannot describe, and adds it to the live state (and to history, when that is on).
- It reads only the changed area of such a window, at most every 2 s per window, within an NPU time budget, with
  the same masking as `ocr_windows`. The text is redacted like any other.
- It runs only with the recognizer on the NPU, never as a constant CPU load; until that build exists it stays
  paused and `scribe_status` says why. With ScribeRec on the NPU (OCR.md), a worst-case canvas of constantly
  changing text cost 1.3% of a core in the server and 1.6% of the machine in the worker.
- It pauses, holding no NPU context, while an ignite-xdna NPU measurement asks for a quiet device (its
  `npu-quiet.<pid>.request` files), on battery, and after `python -m ignite_scribe pause` (until `resume`).
  NPU programs that do not write a quiet request (a direct Ignition run, an ad-hoc probe) are not seen, and
  would share the device with it.

**Vision reads pixels when called, or always-on when that is enabled.**
- `find_ui_elements` and `ocr_windows` copy the on-screen pixels of the windows they look at, in memory, never
  to disk.
- It never copies windows above them, excluded windows, or password fields. Those areas are left grey.
- It skips elevated windows unless the policy sets `capture_elevated`.
- OCR text is redacted like UI Automation text, and lines lying under masked areas are dropped.
- The vision worker starts on the first call and exits after 30 s idle, releasing the NPU.
- `IGNITE_SCRIBE_VISION=0` turns both tools off.

The detector is OmniParser V1's `icon_detect`, under AGPL-3.0; it is not shipped here. Building it for the
NPU needs [ignite-xdna](https://github.com/jdominick05/ignite-xdna); [VISION.md](VISION.md) has the steps.

**Tests.** `python -m pytest -q` from a checkout (offline, plus live Win32 and UI Automation tests on Windows).
