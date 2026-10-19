# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turn OmniParser V1's icon detector (model.yaml + model.safetensors, AGPL-3.0) into an Ultralytics .pt and a
static ONNX for the ignite-xdna graph engine.

Mirrors OmniParser V1's own weights/convert_safetensor_to_pt.py (build the YOLO from model.yaml, load the state
dict), then exports the way ignite-xdna's pipelines/yolov8n/1_export.py does: opset 17, static batch 1, square
imgsz, simplify, no NMS. Run in an env with torch + ultralytics (ignite-xdna's resnet_env).

    python tools/convert_omniparser_v1.py --src models/omniparser_v1/icon_detect --imgsz 640
"""
import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="models/omniparser_v1/icon_detect")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    import ultralytics
    from safetensors.torch import load_file
    from ultralytics import YOLO

    src = Path(a.src)
    model = YOLO(str(src / "model.yaml"))
    missing, unexpected = model.model.load_state_dict(load_file(str(src / "model.safetensors")), strict=False)
    print(f"ultralytics {ultralytics.__version__}; missing keys {len(missing)}, unexpected {len(unexpected)}")
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    head = model.model.model[-1]
    print(f"Detect: nc={head.nc} reg_max={head.reg_max} strides={[int(s) for s in head.stride]} "
          f"params={sum(p.numel() for p in model.model.parameters()):,}")
    model.model.names = {0: "icon"}
    pt = src / "model.pt"
    model.save(str(pt))
    out = YOLO(str(pt)).export(format="onnx", opset=17, imgsz=a.imgsz, simplify=True, dynamic=False, batch=1,
                               nms=False)
    dst = src.parent / f"icon_detect_v1_{a.imgsz}.onnx"
    Path(out).replace(dst)
    print("wrote", pt, "and", dst)


if __name__ == "__main__":
    main()
