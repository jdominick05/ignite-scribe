# OCR: on the CPU now, on the NPU later

`ocr_windows` (0.0.6) reads text from window pixels, with PP-OCRv5 mobile detection and recognition, both on the
CPU inside the vision worker. This page records what was measured, and what still keeps the recognizer off the
NPU. Everything is CPU and NPU on Desktop 2 (Ryzen 7 8700G, Phoenix), 2026-10-19.

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
- The recognizer on the NPU (above). The detector stays on the CPU too: it is a DBNet with ConvTranspose,
  Resize and ten squeeze-excite blocks, none of which the engine runs; not surveyed further.
- Width buckets for fixed-width NPU builds (`ocr_pipeline.bucket`). The CPU model takes any width.
- Accuracy on real screens beyond one terminal window.

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
