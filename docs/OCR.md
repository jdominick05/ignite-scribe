# OCR on the NPU: where it stands

Not shipped. This page records what was measured toward it and what blocks it. Everything is CPU and NPU on
Desktop 2 (Ryzen 7 8700G, Phoenix), 2026-10-19.

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
- Text-line detection: PP-OCRv5's detector is a DBNet with bilinear upsampling, not surveyed yet.
- The OCR stage in the server.
- Any accuracy on real screens.
