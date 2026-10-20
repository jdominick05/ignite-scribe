# OCR: the detector on the CPU, the recognizer on the NPU

`ocr_windows` (0.0.6) reads text from window pixels inside the vision worker. Lines are found by PP-OCRv5's
detector on the CPU. Since release-01 (0.0.9), they are read by ScribeRec, a recognizer trained for the NPU's int8
engine, on the NPU, with PP-OCRv5's recognizer as the CPU path. This page is the record: what was measured, and
in what order. The early sections describe the PP-OCRv5-only stage. Everything is CPU and NPU on Desktop 2 (Ryzen 7 8700G, Phoenix), 2026-10-19.

## Getting the models
Into `models/ocr/` (git-ignored; `IGNITE_SCRIBE_OCR_DIR` points elsewhere). Both models are Apache-2.0.
- **Detector:** `ppocrv5_det.onnx` from `ilaylow/PP_OCRv5_mobile_onnx`, saved as `ppocrv5_mobile_det.onnx`.
- **Recognizer:** `languages/english/rec.onnx` from `monkt/paddleocr-onnx`, saved as `en_ppocrv5_rec.onnx`.
- **Characters:** `dict.txt` from the same folder as the recognizer.

## The pipeline (`ocr_pipeline.py`)
1. **Capture:** the same masked capture as the vision stage.
2. **Detect:** the DBNet detector at the window's resolution, capped at 1920 on the longer side.
3. **Boxes:** axis-aligned, grown by DB's unclip distance. Boxes on one row closer than their height are joined:
   the detector splits a row at every wide space, and each piece costs a recognizer call.
4. **Recognize:** each box in reading order, BGR as the model was trained, at height 48 and a width padded to 32.
5. **Return:** lines map to screen pixels, and lines lying under masked areas are dropped.
6. **Redact:** `Policy.apply` scrubs the result like UI Automation text.

## End to end
- **Synthetic pages:** `tools/ocr_page_eval.py`, 20 pages of 1280x800 with 421 GDI lines.
  - Line recall 0.998, CER 0.0040 over all truth characters, 1 spurious line.
  - 274 ms a page: 109 ms detection, 162 ms recognition.
  - Without row merging: CER 0.0075 and 8 spurious lines.
- **A real window** (`tools/ocr_vs_uia.py`): a 1936x1048 Windows Terminal, whose UI Automation Text pattern is
  ground truth for its pixels. Numbers only; no text is printed or kept.
  - Each OCR line is scored against the closest substring of a UIA line, with whitespace runs collapsed.
  - Three reads: **CER 0.0067-0.0078**, 67-75 % of lines exact, 1.24-1.32 s per call on the CPU.
  - Before row merging, the same check read 252 fragments for 57 rows.
  - The terminal was showing a live session, so its content moved between the two reads.

## The recognizer
PP-OCRv5 mobile, English (`en_PP-OCRv5_mobile_rec`, Apache-2.0; the ONNX conversion from
`monkt/paddleocr-onnx`):
- **Backbone:** PP-LCNetV3, 33 convolutions, mostly depthwise plus pointwise, with HardSwish and two
  squeeze-excite blocks.
- **Tail:** an average pool, an SVTR neck (attention) and a 438-class CTC head.
- **Input:** a text line at height 48, any width, normalized to [-1, 1].

`ignite_scribe.ocr` holds the line transform and greedy CTC decoding (numpy).

## The eval set is synthetic
No public set of desktop screenshots with line transcriptions was found:
- ScreenSpot's labels are instructions;
- the SourceForge screenshot set's labels are page-level VLM output, with no stated licence.

`tools/gdi_lines.py` draws lines the way Windows draws UI text: GDI `DrawTextW` with ClearType, into an
off-screen bitmap. It uses six UI fonts at 12-24 px, on light and dark backgrounds, and reads nothing from the
screen. Every number below is on 500 such lines (seed 100). Calibration lines are a separate seed. **Synthetic
lines are cleaner than real screens, so these are not real-world accuracy figures.**

## Accuracy and CPU time (500 lines)

| reader | CER | lines exact | ms / line (median) |
|---|---:|---:|---:|
| PP-OCRv5 float, each line at its own width | **0.0049** | **91.8 %** | 6.59 |
| PP-OCRv5 float, fixed 48x320 (long lines squashed) | 0.3399 | 64.4 % | 5.64 |
| Windows.Media.Ocr, line upscaled 2x | 0.0190 | 75.2 % | 2.31 |
| Windows.Media.Ocr, line upscaled 3x | 0.0191 | 75.6 % | 3.72 |
| Windows.Media.Ocr, native size | 1.0 (reads nothing) | 0 % | - |

PP-OCRv5 makes about a quarter of Windows OCR's character errors. A fixed width must not squash lines: a container
per width bucket (for example 320 / 640 / 960 / 1280) is the plan.

## What the engine compiles today
Each probe is a one-feature model (`tools/engine_probe_models.py`), compiled with `--task segment
--no-workspace-reuse` and run on the device.
- **Compiles, and the device output matches the engine's integer reference exactly** (0 steps;
  `tools/verify_dense_probe.py`):
  - line shapes 40x320 and 20x160, at 0.22-0.32 ms per dispatch;
  - a 24-row map;
  - depthwise 3x3, HardSwish, and a conv with no activation into a conv;
  - squeeze-excite as a `--host-region` (checked against ORT only).
- **Refused:**
  - maps of 12, 6 or 3 rows: "a 12-pixel map is smaller than one 20-pixel tile";
  - strides (2,1) and (1,2), at 3x3 and 5x5;
  - HardSwish -> scalar Mul -> scalar Add -> depthwise conv, PP-LCNetV3's "learnable affine block": a KeyError.

`tools/fold_lab.py` folds every affine step that folds exactly (CER unchanged). The 13 that remain sit before a
zero-padded k x k conv. Folding those is wrong at the border, and on 3-6-row maps the border is most of the map:
tried, CER 0.996.

## Why int8 cannot carry this model, and bf16 can
CPU, 100 of the lines, fixed 48x320. The float backbone plus tail reads CER 0.341 at this width, the reference
for the rows below.

| backbone arithmetic | CER |
|---|---:|
| float | 0.341 |
| Quark XINT8 (CLE on or off, calibrated on GDI lines) | 0.99 |
| int8 per-tensor power-of-two weights, float activations | 0.731 |
| int8 per-channel power-of-two weights, float activations | 0.347 |
| ORT QDQ: uint8 activations, int8 per-channel weights (free scales) | 0.571 |
| ORT QDQ: uint8 activations, int8 per-tensor weights | 0.972 |
| bf16 activations, bf16 weights (simulated) | 0.352 |
| bf16 activations, int8 per-channel power-of-two weights (simulated) | 0.349 |
| bf16 activations, int8 per-tensor weights (simulated) | 0.732 |

- **Two separate losses:**
  - per-tensor weight scales, which XINT8 and the int8 engine use; the depthwise layers' channels span 4-13x
    in range;
  - uint8 activations.
- **What does not help:** CLE cannot fix it, because equalization does not pass through HardSwish or the affine
  blocks. Quark refuses `per_channel` under XINT8.
- **What does:** with bf16 activations and per-channel (or bf16) weights the model stays at float accuracy.
- `tools/quant_layerwise.py` shows the int8 decline layer by layer. It is gradual, then collapses at the 5x5
  depthwise layers with float ranges of 87 and 176.

**So the route is ignite-xdna's bf16 datapath** (W8A16, branch `worktree-bf16-engine`), extended to this
backbone's maps under 20 rows, asymmetric strides and the affine block. The bf16 simulation rounds with casts; it
is not the engine's accumulate model, so the device would still need measuring. These requests went to the
ignite-xdna session with repros. They are engine work, and not started.
The other route is an engine-native recognizer, trained with quantization in the loop: a GPU job for Desktop 1.

## Not done
- ~~The recognizer on the NPU~~: done, ScribeRec release-01 (below).
- ~~Width buckets for fixed-width NPU builds~~: done, four buckets (320 / 640 / 960 / 1280).
- **The detector stays on the CPU** (about 62 ms per always-on read). It is a DBNet with ConvTranspose, Resize and
  ten squeeze-excite blocks, none of which the engine runs; not surveyed further.
- **ScribeRec on real screens is worse than PP-OCRv5** (below): it was trained on synthetic UI text only.
- **Installed copies cannot get the model yet:** release-01 and its containers live in git-ignored folders here.

## The route chosen: an engine-native recognizer (2026-10-19)
The user chose to train a recognizer shaped for the int8 engine rather than extend the engine for PP-OCRv5. The
ignite-xdna session estimated PP-OCRv5's backbone at about 130 ms a line on the bf16 engine (DERIVED). Its
depthwise convolutions are lowered dense (20x the MACs) and its 3/6/12-row maps pad to 20-row tiles (65x).

`tools/scriberec.py` defines the candidate:
- an 80-row input, two square stride-2 stages, then everything at 20 rows;
- dense 3x3 convolutions with residual adds and ReLU, channels in multiples of 32;
- a CPU tail: a learned weighting over the 20 rows, a 2-layer BiLSTM, and CTC.

Random weights, 80x320, Quark XINT8 (noise calibration), `--task segment`, on the NPU:

| backbone | MACs | layers on the NPU | exact vs reference | dispatch, overlap (default) | no overlap |
|---|---:|---:|---|---:|---:|
| 32/64/128, blocks 1/2/3 | 1.95 G | 16 of 16 | 0 steps | 1.58 ms | 1.83 ms |
| 32/64/96, blocks 1/1/2 | 0.90 G | 12 of 12 | 0 steps | 0.92 ms | 1.08 ms |

The CPU tail takes 0.96-1.07 ms with 4 threads (ORT, 100 runs). A line therefore costs about 1.9 ms (small) or
2.6 ms (big), against about 6.3 ms for PP-OCRv5 on the CPU. Accuracy is unknown until it is trained. Training
runs on Desktop 1's 7900 XTX (the "Model Work" session). The trained model's recipe goes to ignite-xdna, which
compiles and verifies it.

## Always-on OCR: what one read costs (2026-10-19, Desktop 2)
`tools/region_cost.py` uses synthetic pages (GDI lines, nothing from the screen) cropped to the sizes a changed
region takes. The pipeline is PP-OCRv5 on the CPU, 4 threads; values are medians of 15 crops, on a quiet host.

| changed region | detect | recognize | lines |
|---|---:|---:|---:|
| 1000x60 (a log line) | 6.2 ms | 4.6 ms | 1 |
| 1000x120 | 13.6 ms | 12.0 ms | 2 |
| 1000x250 | 23.6 ms | 31.2 ms | 4 |
| 1280x400 | 49.1 ms | 59.7 ms | 9 |
| 1000x700 (a whole window) | 74.5 ms | 82.5 ms | 12 |
| 1920x1040 (the whole screen) | 263 ms | 214 ms | 28 |

What follows from the table (DERIVED):
- A log or chat window adding a line or two, read every 2 s, costs about 7 ms of detection a read, under 1% of a
  core, so the CPU detector is acceptable there.
- A window that repaints entirely (video, a game) costs 75-263 ms of detection a read. At one read per 2 s that is
  4-13% of the 4 threads, and the per-minute budget caps it.
- The recognizer's share moves to the NPU with ScribeRec.

A detector shaped for the engine is worth training only if full-window repaints turn out to be common.

The live run on the real desktop read nothing in 5 minutes: both windows were described by UI Automation. A
synthetic canvas window (`tools/canvas_window.py`) opened behind the focused window, 100% covered, so its
masked capture read nothing too, which is the masking working as intended.

**Live, with a visible synthetic canvas (2026-10-19, Desktop 2, CPU measurement mode, no NPU; the user agreed to
a topmost window for 3 minutes).** `tools/always_on_live.py --canvas --seconds 180` (budget lifted, 2 s gap):

| reads/min | pixels per read | lines per read | detect | recognize | worker busy | server process CPU |
|---:|---:|---:|---:|---:|---:|---:|
| 24.7 | 700 k (the whole 1000x700 window) | 27 | 80 ms | 326 ms | 17% of the wall time | 2.0% |

- The canvas repaints its whole area each second, as a scrolling log, a chat or a video does. So every changed
  rectangle was the whole window, and every read recognized every line again. The "a line or two" case in the
  table above holds only for content that changes in place.
- The recognizer dominates (326 of 406 ms). On the NPU, ScribeRec's measured 0.92 ms backbone plus about 1 ms of
  tail gives about 50 ms for 27 lines (DERIVED).
- The next saving is to skip lines that did not change: hash each detected line's pixels and reuse its text. For
  scrolling content that is nearly every line.
- The default budget (6 s of OCR a minute) would cap this window at about 15 reads a minute.

**Reusing unchanged lines (`ocr_pipeline.line_key`).** Each detected line is keyed by a digest of its ink: the
crop trimmed to the pixels that differ from its background, so the key holds however loosely the box sits and
wherever the line has scrolled to. A line already read is not recognized again. Lines touching the region's edge
(possibly cut off) and reads under 0.9 confidence are not cached. Only digests and text are kept, in the worker's
memory, at most 2048 lines. `tools/scroll_cache.py` scrolls synthetic pages through a 1000x700 view, 37 px a
frame, 40 frames, on the CPU:

| page seed | lines reused | recognize per frame, cached / fresh | CER, cached / fresh |
|---|---:|---|---|
| 91 | 89.1% | 12.6 / 153.2 ms | 0.00184 / 0.00161 |
| 92 | 88.9% | 12.5 / 128.3 ms | 0.00177 / 0.00149 |
| 93 | 89.9% | 12.9 / 140.7 ms | 0.00078 / 0.00149 |

The CER is the same on average (0.0015 both ways). A cached read keeps the first read's result, right or wrong,
while fresh reads vary from frame to frame, so scrolling text also stops flickering in change events.

## ScribeRec at inference (`scriberec_rec.py`), ready for the trained weights

The worker side follows the training code's input contract exactly (confirmed with the trainer):
- PIL `BILINEAR` to height 80, `nw = max(1, round(w * 80 / h))`, `/ 255`;
- right-pad by repeating the last column to `W32 = max(32, ceil(nw / 32) * 32)`, then to the bucket (320, 640,
  960, 1280: one static backbone each);
- greedy CTC over the first `W32 / 4` columns only, since bucket padding was never in a training target;
- lines wider than 1280 after the resize were never trained, so they are cut at the lowest-ink column near each
  boundary and the pieces joined.

The tail (BiLSTM and CTC head, CPU) runs batched over the lines of one bucket. That is exact, because they share
its T (0.0 difference measured), and cheaper: 0.34-1.42 ms a line at batch 8 against 0.93-3.61 ms one at a time.

A hand-back directory (`models/scriberec/`: `tail.onnx`, `backbone_<W>[_xint8].onnx`) plus `build/scriberec_<W>.ignite`
for all four buckets is what always-on waits for. Its readiness check needs every bucket on the NPU, since a CPU
bucket would make the backend `mixed`, which always-on refuses.

Plumbing checked with random weights (seed 5, SMALL: widths 32/64/96, blocks 1/1/2, 96 out), compiled for all
four buckets (12 layers each, none on the host), `tools/scriberec_parity.py`, Desktop 2, quiet host, 8 synthetic
lines per width:

| resized width | bucket | NPU vs XINT8 ONNX, worst | same text | ms per line, NPU / CPU int8 |
|---:|---:|---:|---:|---|
| 200 | 320 | 0 steps | 8/8 | 12.3 / 18.5 |
| 500 | 640 | 0 steps | 8/8 | 14.1 / 22.4 |
| 800 | 960 | 0 steps | 8/8 | 15.9 / 28.2 |
| 1150 | 1280 | 0 steps | 8/8 | 23.8 / 41.2 |
| 1900 | split | 0 steps | 8/8 | 34.3 / 60.0 |

Per line on the NPU path, one line per call:

| stage | ms |
|---|---:|
| prepare | 0.89 |
| backbone call, total | 9.6 |
| of which input quantize and write | 6.2 |
| of which dispatch | 6.8 |
| of which readback | 2.9 |
| tail | 5.3 |
| decode | 0.36 |

The backbone figures are summed over pieces. An earlier run of the same check, before `read_many`, was about 15% faster throughout.

**Where this stands against PP-OCRv5 on the CPU.** On one 1000x700 synthetic page, about 20 boxes, the NPU path spent 220-237 ms recognizing against PP-OCRv5's 150 ms. It is not faster per line yet, and it still costs CPU per line: about 6 ms to quantize and write the input, and 5 ms of tail. What it changes is where the work runs, which is what always-on needs. Three things should cut the host share before always-on ships, each to be measured on the trained model:
- the line cache;
- the batched tail;
- handing PIL's uint8 straight to the container. This works only if its input scale is 1/255 with zero point 0, and must be checked bit-equal first.

**Stop paths:** `tools/worker_lifecycle.py --ocr`, with the four bucket sessions open next to the detector:

| stop path | NPU idle after |
|---|---:|
| `close()` | 0.08 s |
| idle stop (3 s idle) | 0.07 s |
| parent killed | 0.06 s |

A failed bucket open (device held) closes the buckets already open, and the recognizer falls back to the CPU, reporting `ocr_reason`. With `--backend npu` it raises instead. Always-on refuses either way, since the backend is not `npu`.

**Installing the trained model:** copy the hand-back directory (`ignite-xdna/models/scriberec/<run>/`) to `models/scriberec/`, or point `IGNITE_SCRIBE_OCR_REC_MODEL` at it. Put the four compiled buckets at `build/scriberec_{320,640,960,1280}.ignite` (or set `IGNITE_SCRIBE_OCR_REC_CONTAINERS`).

End to end through the real worker (detector and all four buckets open at once), it replies `ocr_backend: npu`,
and xrt-smi shows no contexts after `close()`.

## ScribeRec release-01 (trained)

Trained by Model Work on Desktop 1. The training text was public domain only (PG19, SQLite source, and the line
generators). Checkpoint: the last step (26000) of the cosine schedule, not chosen by eval. The hand-back is in
`ignite-xdna/models/scriberec/release-01/`, and the SHA-256 of each `.onnx` was checked on arrival.

On the same 500 seed-100 lines as the table above, rendered on Desktop 2, through the shipped inference path
(`tools/scriberec_eval.py`, CPU, batch 8):

| recognizer | CER | exact lines | ms per line (CPU) |
|---|---:|---:|---:|
| ScribeRec release-01, four NPU buckets (`build/scriberec_<W>.ignite`) | **0.00181** | **97.0 %** | 9.1 |
| ScribeRec release-01, XINT8 backbones | **0.00181** | **97.0 %** | 18.4 |
| ScribeRec release-01, float backbones (before the split fix) | 0.00199 | 96.6 % | 13.7 |
| PP-OCRv5 float, own width (above) | 0.0049 | 91.8 % | 6.59 |

- Quantization costs nothing here: XINT8 and float give the same output.
- The 494 lines up to 1280 px after resize: 20 edits, CER 0.00186. The trainer, rendering seed 100 on
  Desktop 1, got the same 20 edits, so the two machines' ClearType output agrees.
- The 6 wider lines are split, since no trained width fits them. The first split cut at the lowest-ink column,
  and twice that fell between two letters and was joined with a space: 2 edits, both an extra space. It now
  cuts in the middle of the widest blank run within 320 px of the boundary, and adds a space only for a run at
  least 0.1 x the line height wide. Result: 0 edits, the same as the trainer decoding them whole at 1600 wide.
  That brings the total to the trainer's 0.00181 / 97.0%.
- Release-01 sometimes reads a Latin capital as its Greek twin, for example Chi for X. `scriberec_rec`
  maps the 14 twins back (`latin_twins`, on by default) until a fine-tune fixes it. No seed-100 label
  contains Greek.
- **On the NPU:**
  - The four buckets come from ignite-xdna's `recipes/scriberec.toml`: 12 layers each, none on the host, and
    each manifest's source hash matches its XINT8 backbone.
  - They give the same CER and exact share as the CPU XINT8 path, at 9.1 ms a line (18.4 on the CPU), measured
    through `research-lowlevel.sh --npu` on a clear host.
  - `tools/worker_lifecycle.py --ocr` with release-01 loaded: `close()`, idle stop and parent death each leave
    the NPU idle within 0.06 s.
  - The DMA completion tokens balance in the instruction streams of all four buckets and of the four
    `icon_detect_v1*.ignite` builds: every token issued is waited for, on every shim channel of every column.
    ignite-xdna checked this statically on 2026-10-19, after an unbalanced stream in its LLM work kept wedging
    the device. Always-on dispatches these streams continuously. The check reads the stored streams only, not
    anything patched in at dispatch.
- Lines are synthetic, so these are not real-world figures.

## Always-on as shipped, live (NPU recognizer, line cache)

`tools/always_on_live.py --npu --canvas --seconds 180`, Desktop 2, 2026-10-19, run through `research-lowlevel.sh
--npu` on a clear host. The canvas is the topmost synthetic GDI window, about 700 kpx of text that changes
continuously, which makes it a worst case. Numbers only.

| | CPU recognizer (earlier run) | as shipped |
|---|---:|---:|
| reads per minute | 24.7 | 27.0 |
| detect ms per read (CPU) | 80 | 61.9 |
| recognize ms per read | 326 | 114.0 |
| lines reused from the cache | none | 1544 of 1983 (78%) |
| share of each minute busy | 17% | 8.3% |
| MCP server process CPU | 2.0% of a core | 1.3% of a core |
| vision worker CPU, startup included | not measured | 25.6% of a core (1.6% of 16 threads) |

- Errors: 0. The backend was `npu` for the whole run, and the NPU was idle after the stop.
- The text detector still runs on the CPU, and at 4 threads it is most of the worker's CPU time.
- This is the canvas at the maximum read rate (one read per window per 2 s). A desktop with less changing text
  costs proportionally less, and an unchanged screen costs nothing, since reads follow dirty regions.

## ScribeRec on a real window (2026-10-19, Desktop 2)

`tools/ocr_vs_uia.py`: a Windows Terminal (Cascadia Mono, dark theme), whose UI Automation Text pattern is ground
truth for its pixels. Numbers only; no text was printed or kept. The three arms ran back to back in one sitting,
twice, through `research-lowlevel.sh --npu`: clear host, timing-eligible, the NPU idle afterwards.
Log: `results/ocr_vs_uia_scriberec_232953.log`.

| recognizer | CER vs UIA | exact lines | recognize ms |
|---|---:|---:|---:|
| PP-OCRv5 (CPU) | **0.0095 / 0.0097** | 67.8% / 60.4% | 284 / 47 |
| ScribeRec release-01 (CPU) | 0.044 / 0.030 | 46.6% / 41.5% | 545 / 116 |
| ScribeRec release-01 (NPU) | (0.675) / 0.030 | (9.1%) / 41.5% | (1539) / 50 |

- **On the NPU it is exact:** in round 2 its output equals the CPU's.
- **Round 1's NPU arm is not counted:** the terminal changed between its two reads (UIA lines 45 -> 39). It also
  included the first NPU open.
- **On real text, ScribeRec makes 3-4.5x the errors of PP-OCRv5,** although it made 2.7x fewer on synthetic lines.

**The breakdown,** from two more CPU runs of each (numbers only). The terminal was live, so a third pair was
contaminated (CER 0.67, with the line counts drifting) and is not counted.
- **Spacing:** ignoring spaces changes neither recognizer's CER, so spacing is not the cause.
- **Plain-ASCII lines:** ScribeRec 0.026, PP-OCRv5 0.006.
- **Non-ASCII output:** ScribeRec emits it in 20 lines where PP-OCRv5 has 4-8. This fits its Greek-twin habit
  and other lookalikes the twin map does not cover.
- ~~Likely cause: Cascadia Mono is not among the training faces.~~ **Wrong:** that list is the *eval-set*
  generator's (`tools/gdi_lines.py`). ScribeRec's training generator has Cascadia Mono and Cascadia Code, every
  Cascadia weight, 30% dark-theme lines and 15% coloured text (Model Work, 2026-10-19).

**The diagnosis** (`tools/ocr_diag.py`, CPU; aggregate only, no text or crops kept):
- **Method:** a live terminal, three samples, 226 lines. Truth is the lines present in UI Automation both before
  and after each capture. All three recognizers read the same detector crops.
- **CER:** ScribeRec as shipped 0.079; ScribeRec with every class outside printable ASCII masked before the argmax
  0.0738; PP-OCRv5 0.0214. That terminal held UI glyphs (⏵, ·, ✔), so CERs are higher than in the table above, but
  the ratio holds (3.7x).
- **Charset masking is not the fix:** it removes about 7% of ScribeRec's errors.
- **Spaces are the biggest error class:**
  - a space read as `-` 47 times, and dropped 47 times;
  - a space read as `'`, `¸` or `"` 31 times;
  - then lowercase confusions: l -> i 13, o -> e 9, u -> t 9, c -> e 6.
- **Crop heights** before the resize to 80 are mostly 28-39 px (CER 0.05-0.09); below 20 px it rises to
  0.26-0.33.
- **Crops come from the detector's ink boxes, not UIA ranges.** The median ink padding is 2 px above and below,
  but 46% of crops have none above and 38% none below. Training lines have 0-6 px.
- **The terminal:** Windows Terminal at its defaults (Cascadia Mono 12 pt, grayscale antialiasing, AtlasEngine,
  i.e. DirectWrite), at 120 DPI (125%). Every training line is GDI `DrawTextW` with ClearType.

**What follows.**
- PP-OCRv5's recognizer is the more accurate one on real screens today. Always-on OCR still needs ScribeRec,
  since it runs on the NPU only.
- **The leading suspects** for release-02 (Model Work's, consistent with the above):
  - the renderer: a DirectWrite path with grayscale antialiasing in the generator;
  - crop geometry: more zero-padding crops, matching the detector's boxes;
  - space handling at monospace cell gaps.
- `ScribeRec(charset=...)` exists for the mask, which is off by default.

## ScribeRec release-03, installed (2026-10-20, Desktop 2)

Release-02 and r02b failed on real terminals, worse than release-01 on the same crops. The cause was the crop,
not the font (`tools/ocr_diag.py`, aggregates only):
- terminal rows have no leading, and the detector's boxes are 1.0-1.3 rows tall, so the neighbouring rows' ink is
  in most crops;
- on a static terminal, 91% of release-01's wrong lines have ink in their top 3 px rows, against 70% of all
  lines;
- its top errors were marks that sit high on a line (a space read as ' or ¯): the descenders of the row above.

Release-03 (Model Work, a fine-tune of release-01) trains on that geometry. It stacks three rows at the cell
pitch with zero leading, crops 0.8-1.4 pitches around the target row, and adds prompt, cursor and box glyphs at
the edges.

**The gate** (`ocr_diag.py --static docs/DESIGN.md`): a new terminal window showing a tracked file, so its UI
Automation truth cannot move. All recognizers read the same crops. Pass mark: at most 1.5x PP-OCRv5's CER.

| recognizer | static terminal, CER (3 runs) | live terminal TUI, CER |
|---|---:|---:|
| PP-OCRv5 (CPU) | 0.0048-0.0054 | 0.0153 |
| ScribeRec release-01 | 0.0140-0.0155 | 0.0829 |
| ScribeRec r02b | 0.0198-0.0209 | 0.0439 (other run) |
| **ScribeRec release-03** | **0.0022** | 0.0447 |

- **On the NPU** (XDNA Worker, `tools/scriberec_parity.py --truth`): worst 0 steps against the XINT8 ONNX in all
  buckets and splits, 12/12 lines the same, CER 0.00151 on both paths. Per line it costs 10.4-16.9 ms
  (29.8 ms for a split line), against 18.5-50.0 ms for CPU int8.
- **The live TUI is still the weak case:** a space is read as '-' next to box-drawing rules, and spinner and
  check glyphs (◐ ✔ ◻) that aren't in the dictionary get inserted.
- **Installed:** `models/scriberec` with `build/scriberec_<W>.ignite`. Release-01 is kept as
  `models/scriberec_r01` and `build/scriberec_r01_<W>.ignite`.
