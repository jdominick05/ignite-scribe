# Vision stage: the detector and its scale

## The model
OmniParser **V1** `icon_detect` (`microsoft/OmniParser`, `icon_detect/`, AGPL-3.0): a 1-class (`icon`) YOLOv8n,
3,005,843 parameters, DFL with 16 bins, strides 8 / 16 / 32. `tools/convert_omniparser_v1.py` builds it from
`model.yaml` + `model.safetensors` (OmniParser's own conversion) and exports a static 640x640 ONNX (opset 17).
ignite-xdna's `pipelines/yolov8n/1b_cut_head.py` cuts it to the six raw head convolutions.

Why not V2: V2's `icon_detect/model.pt` is a 1-class **YOLO11m** trained at 1280. Its sidecar `model.yaml` /
`train_args.yaml` describe a YOLOv8n and are stale; the checkpoint's own embedded config says `yolo11m.yaml`.
The ignite-xdna graph engine builds YOLO only at square 640 today. Its YOLO11 needs a host attention segment.
V1's YOLOv8n is the engine's best-proven shape.

## Decode
`ignite_scribe.vision.decode_heads` reproduces the cut-off tail: softmax over 16 bins, expected side distances,
anchors at cell centres, sigmoid score. On a 640 letterboxed screenshot it equals Ultralytics' in-graph decode:
857 of 857 boxes above 0.05, max box error 0.0001 px, max score error 0.

## Scale study (CPU, float weights)
`tools/scale_study.py`, over the 231 desktop screenshots of ScreenSpot-v2 (Apache-2.0, public; never the user's
screen), 334 labelled targets (140 icon, 194 text). One sitting on Desktop 2.
- **Metric:** the share of targets covered by a detection, at IoU >= 0.5 and by centre inside a box.
- **Settings:** NMS IoU 0.45, conf 0.05 (OmniParser's box threshold).

**By mode, all images:**

| mode | inferences / image | icon IoU50 | icon centre | text IoU50 | text centre |
|---|---:|---:|---:|---:|---:|
| whole screen letterboxed to 640 | 1.00 | 0.721 | 0.936 | 0.753 | 0.907 |
| tiles at 0.5 | 2.22 | 0.736 | 0.957 | 0.758 | 0.923 |
| tiles at 0.75 | 3.92 | 0.721 | 0.957 | 0.814 | 0.933 |
| tiles at 1.0 | 6.21 | 0.757 | 0.957 | 0.747 | 0.918 |
| tiles at 1.5 | 13.23 | 0.707 | 0.929 | 0.649 | 0.861 |

Most of these images are small: 109 of 231 are 960x540. That makes the mode averages close, and the variable
that matters is **the element's size at the detector input**. Pooled over modes (icon IoU50 / centre):

| size at input (px, sqrt of area) | 8-12 | 12-16 | 16-24 | 24-32 | 32-48 | 48-64 | 64-96 | 96+ |
|---|---|---|---|---|---|---|---|---|
| icon | 0.20 / 0.60 | 0.20 / 0.90 | 0.72 / 0.96 | 0.83 / 0.98 | 0.79 / 0.97 | 0.83 / 0.98 | 0.75 / 0.98 | 0.54 / 0.84 |

On the 2560x1440 and 2880x1800 screenshots alone (14 icon targets per mode, a small sample), letterboxing the
whole screen to 640 drops icon IoU50 to 0.36: icons shrink to about 19 px. Tiles at 0.5 read 0.64 there (median
42 px at the input).

**Decision:** tile the regions that need vision at **scale = 96 / the window's DPI**. At this machine's 100%
scaling that is native tiles, which keeps typical 16-32 px icons inside the band the detector handles. At 200%
it is half scale. Tiles are 640, overlapping by 96 px. Costs, worst case:
- a full 2560x1440 screen at scale 1.0 is 15 tiles;
- only windows selected for vision are tiled, not the whole screen.

At conf 0.25 every mode loses about a third of its recall (icon IoU50 0.38-0.46), so the default stays at 0.05.

## Building it for the NPU
Paths are relative to this repo. Each step runs in the env named, from an ignite-xdna checkout:
1. `hf download microsoft/OmniParser --include "icon_detect/*" --local-dir models/omniparser_v1`
2. `tools/convert_omniparser_v1.py` (resnet_env: torch + ultralytics) writes `icon_detect_v1_640.onnx`.
3. ignite-xdna `pipelines/yolov8n/1b_cut_head.py --in ... --out ..._cut.onnx` (resnet_env).
4. `tools/make_calib_tiles.py` writes 64 calibration tiles from ScreenSpot-v2's even-positioned desktop
   screenshots, cut by the runtime's own tiler at scales 1.0 and 0.5.
5. ignite-xdna `pipelines/yolov8n/3b_quantize_cut.py --in ..._cut.onnx --calib-dir calib/tiles640 --limit 64
   --out ..._cut_xint8.onnx` (resnet_env, AMD Quark XINT8).
6. `bash scripts/research-iron.sh -m ignite_xdna.compiler.cli --engine graph --input ..._cut_xint8.onnx
   --output build/icon_detect_v1.ignite`, **without** `--silu-sigmoid` (see Accuracy below). All 66 layers lower
   to the NPU, 0 on the host. The container is 8.4 MB.
   - The worker's CPU fallback is the XINT8 ONNX itself: this container computes it bit for bit.
   - For a `--silu-sigmoid` build, `tools/make_sigmoid_reference.py` writes the matching CPU model
     (`..._xint8_sigref.onnx`).

## Exactness (Desktop 2, NPU idle and no foreign context, 2026-10-19)

Two sittings, with the other session's NPU run in between.

**Layer by layer:** ignite-xdna's `tools/verify_engine_container.py` on `--no-workspace-reuse` builds, input a
held-out ScreenSpot tile.

| build | layers exact (against the compiler's emulator) | dispatch, 20 runs |
|---|---|---|
| shipped (engine-default SiLU) | **66 / 66** | 5.61 ms mean (5.55-5.79) |
| `--silu-sigmoid` | **66 / 66** | 7.01 ms mean (6.77-7.33) |

**NPU heads against ONNX Runtime:** `tools/worker_parity.py`, 32 held-out tiles.

| container | reference ONNX | result | NPU per tile |
|---|---|---|---|
| **shipped** | plain XINT8 | **0 steps of difference on every head, 32 / 32 detection sets identical** | 5.63 ms median (5.54-6.05) |
| `--silu-sigmoid` | plain XINT8 | **not equal**: up to 143 steps, 0 / 32 identical. That ONNX computes exact SiLU, the container the core's sigmoid epilogue. | n/a |
| `--silu-sigmoid` | `..._sigref.onnx` | 0 steps, 32 / 32 identical | 6.74 ms median (6.68-6.83) |

## Accuracy: `--silu-sigmoid` costs recall here, so the shipped build does not use it
CPU, the held-out (odd) half of ScreenSpot-v2 desktop: 115 images, 60 icon and 103 text targets, conf 0.05.

| model | mode | icon IoU50 | icon centre | text IoU50 | text centre | detections / image |
|---|---|---:|---:|---:|---:|---:|
| float | tiles@0.5 | 0.783 | 0.967 | 0.757 | 0.893 | 56.2 |
| float | tiles@1.0 | 0.750 | 0.967 | 0.709 | 0.893 | 76.3 |
| XINT8, exact SiLU | tiles@0.5 | 0.733 | 0.967 | 0.709 | 0.922 | 65.1 |
| XINT8, exact SiLU | tiles@1.0 | 0.767 | 0.933 | 0.757 | 0.913 | 93.9 |
| XINT8 + sigmoid SiLU (= the NPU) | tiles@0.5 | 0.567 | 0.783 | 0.641 | 0.854 | 43.7 |
| XINT8 + sigmoid SiLU (= the NPU) | tiles@1.0 | 0.633 | 0.850 | 0.650 | 0.845 | 64.8 |

Quantizing to XINT8 costs little. The sigmoid-epilogue SiLU costs about 0.12-0.22 of icon recall.

**On the NPU** (`tools/scale_study.py --container`, same held-out set):
- the shipped container reads **exactly** the XINT8 row above in every column: tiles@0.5 0.733 / 0.967 /
  0.709 / 0.922 with 65.1 detections per image, tiles@1.0 0.767 / 0.933 / 0.757 / 0.913 with 93.9;
- the `--silu-sigmoid` container reads exactly its sigref row.

So the shipped container is the default-SiLU build: it keeps XINT8's accuracy and is about 1.1 ms faster per
tile. ignite-xdna measured `--silu-sigmoid` as an *improvement* for YOLOv8s COCO mAP (37.21 -> 42.37). That
measurement and this one disagree, and the cause is unexplained here. Candidates:
- a different model and data (1 class, UI screenshots against 80 COCO classes);
- the sigmoid fit depends on each SiLU's quantization scales, which come from this calibration.

## The worker and the stage, live (Desktop 2, 2026-10-19)
- **Lifecycle** (`tools/worker_lifecycle.py`, from this repo's Python 3.14 venv; the worker runs in
  `mlir-aie-iron`, 3.13; one grey and 15 seeded-noise tiles):
  - backend `npu`; 16 tiles took 221.8 ms in one call, 13.9 ms per tile against 5.56 ms on the NPU. The rest is
    the pipe (1.2 MB a tile), int8 conversion, decode and NMS, not yet optimized;
  - first call including the worker's start: 413 ms;
  - **every stop path released the NPU.** `close()`: no hardware context 0.06 s later, with the parent still
    alive. Idle stop (`idle=3`): it exited after 4.1 s. Parent killed: the worker exited within 0.15 s.
- **End to end** (`find_ui_elements` through the MCP stdio server):
  - aimed at a 1936x1048 terminal window: 8 tiles at scale 1.0, nothing masked;
  - about 101 detections, of which UI Automation already covered all but 5;
  - detector 44.7-46.3 ms on the NPU, 141 ms total per call warm, 526 ms for the first (worker start);
  - without `match`, no window on this desktop needed vision (UI Automation described them all).

Not measured: recall on this machine's own windows (by design no screenshots of the user's screen are kept to
test on), and multi-monitor or rotated setups.
