# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The engine side of the detector: start ``npu_worker`` on demand, feed it tiles, stop it when idle.

The worker holds an NPU hardware context while it runs, and this engine runs inside every MCP client session.
So the worker is started only when a vision call needs it, and stopped after ``idle`` seconds without one (30 by
default). Stopping closes its stdin: the worker closes its session, which releases the context, and exits. It
also exits when this process dies, since its stdin then closes.

Configuration, from the environment:
- ``IGNITE_SCRIBE_NPU_PYTHON``: a Python that can import ignite-xdna's runtime (3.13 with PyXRT).
  Default: the ``mlir-aie-iron`` conda env's python.
- ``IGNITE_SCRIBE_XDNA``: the ignite-xdna checkout whose ``src`` goes on the worker's path (default: a sibling
  ``ignite-xdna`` directory).
- ``IGNITE_SCRIBE_CONTAINER`` / ``IGNITE_SCRIBE_ONNX``: the ``.ignite`` container and its CPU reference ONNX.
  Defaults are under this checkout's ``build/`` and ``models/``.
- ``IGNITE_SCRIBE_VISION_BACKEND``: ``auto`` (NPU, else CPU), ``npu`` or ``cpu``.
- ``IGNITE_SCRIBE_OCR_DIR``: the directory with ``ppocrv5_mobile_det.onnx``, ``en_ppocrv5_rec.onnx`` and
  ``dict.txt`` (default: this checkout's ``models/ocr``).
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path.home() / "miniforge3" / "envs" / "mlir-aie-iron" / "python.exe"
DEFAULT_CONTAINER = ROOT / "build" / "icon_detect_v1.ignite"
DEFAULT_ONNX = ROOT / "models" / "omniparser_v1" / "icon_detect_v1_640_cut_xint8.onnx"


def config() -> dict:
    env = os.environ.get
    xdna = Path(env("IGNITE_SCRIBE_XDNA", str(ROOT.parent / "ignite-xdna")))
    ocr_dir = Path(env("IGNITE_SCRIBE_OCR_DIR", str(ROOT / "models" / "ocr")))
    return dict(python=env("IGNITE_SCRIBE_NPU_PYTHON", str(DEFAULT_PYTHON)), xdna=str(xdna),
                container=env("IGNITE_SCRIBE_CONTAINER", str(DEFAULT_CONTAINER)),
                onnx=env("IGNITE_SCRIBE_ONNX", str(DEFAULT_ONNX)),
                backend=env("IGNITE_SCRIBE_VISION_BACKEND", "auto"),
                ocr_det=str(ocr_dir / "ppocrv5_mobile_det.onnx"), ocr_rec=str(ocr_dir / "en_ppocrv5_rec.onnx"),
                ocr_dict=str(ocr_dir / "dict.txt"))


def ocr_available(cfg: dict | None = None) -> tuple[bool, str]:
    """-> (can the worker read text, why not). Checks files only."""
    cfg = cfg or config()
    ok, why = available(cfg)
    if not ok:
        return ok, why
    for key in ("ocr_det", "ocr_rec", "ocr_dict"):
        if not Path(cfg[key]).is_file():
            return False, f"{key} not found: {cfg[key]} (docs/OCR.md: getting the models)"
    return True, ""


def available(cfg: dict | None = None) -> tuple[bool, str]:
    """-> (can a worker be started, why not). Checks files only; starts nothing."""
    cfg = cfg or config()
    for key in ("python", "onnx"):
        if not Path(cfg[key]).is_file():
            return False, f"{key} not found: {cfg[key]}"
    if cfg["backend"] != "cpu" and not Path(cfg["container"]).is_file():
        return False, f"container not found: {cfg['container']}"
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False, "numpy is not installed (pip install -e .[vision])"
    return True, ""


class VisionWorker:
    def __init__(self, idle: float = 30.0, cfg: dict | None = None):
        self.cfg, self.idle = cfg or config(), idle
        self.proc: subprocess.Popen | None = None
        self.backend = None
        self.reason = None
        self.started = self.last_used = 0.0
        self.calls = self.tiles = self.starts = 0
        self._lock = threading.Lock()
        self._timer: threading.Thread | None = None

    # ---- process lifetime ----------------------------------------------------------------------------------
    def _start(self):
        c = self.cfg
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(c["xdna"]) / "src"), str(ROOT / "src")]),
                   PYTHONIOENCODING="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        ocr = [f"--ocr-{k}={c['ocr_' + k]}" for k in ("det", "rec", "dict") if Path(c.get("ocr_" + k, "")).is_file()]
        self.proc = subprocess.Popen([c["python"], "-m", "ignite_scribe.npu_worker", "--container", c["container"],
                                      "--onnx", c["onnx"], "--backend", c["backend"], *ocr],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     env=env, creationflags=flags)
        self.started = time.monotonic()
        self.starts += 1
        h, _ = self._call(dict(op="ping"))
        if not h.get("ok"):
            self._stop()
            raise RuntimeError(f"vision worker did not start: {h}")
        self.backend, self.reason = h.get("backend"), h.get("reason")
        if self._timer is None or not self._timer.is_alive():
            self._timer = threading.Thread(target=self._idle_loop, name="ignite-scribe-vision-idle", daemon=True)
            self._timer.start()

    def _stop(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.stdin.close()                     # EOF: the worker closes its session and exits
            p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            p.kill()
            p.wait(timeout=10)

    def _idle_loop(self):
        while True:
            time.sleep(1.0)
            with self._lock:
                if self.proc is None:
                    return
                if time.monotonic() - self.last_used > self.idle:
                    self._stop()
                    return

    def close(self):
        with self._lock:
            self._stop()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ---- protocol --------------------------------------------------------------------------------------------
    def _call(self, header: dict, raw: bytes = b"") -> tuple[dict, bytes]:
        body = json.dumps(header).encode("utf-8") + b"\n" + raw
        self.proc.stdin.write(struct.pack("<I", len(body)) + body)
        self.proc.stdin.flush()
        n = self.proc.stdout.read(4)
        if len(n) < 4:
            raise RuntimeError("vision worker exited")
        reply = self.proc.stdout.read(struct.unpack("<I", n)[0])
        i = reply.index(b"\n")
        return json.loads(reply[:i].decode("utf-8")), reply[i + 1:]

    def detect(self, tiles, conf: float = 0.05, iou: float = 0.45) -> dict:
        """640x640x3 uint8 RGB tiles -> the worker's reply (per-tile boxes in tile pixels, scores, ms)."""
        import numpy as np
        arr = np.ascontiguousarray(np.stack(tiles).astype(np.uint8))
        with self._lock:
            if not self.running:
                self._start()
            self.last_used = time.monotonic()
            h, _ = self._call(dict(op="detect", n=len(arr), conf=conf, iou=iou), arr.tobytes())
            self.last_used = time.monotonic()
            self.calls += 1
            self.tiles += len(arr)
        if not h.get("ok"):
            raise RuntimeError(f"vision worker: {h.get('error')}")
        return h

    def ocr(self, images) -> dict:
        """HxWx3 uint8 RGB images (any sizes) -> the worker's reply (per-image lines in image pixels)."""
        import numpy as np
        arrs = [np.ascontiguousarray(np.asarray(i, np.uint8)) for i in images]
        with self._lock:
            if not self.running:
                self._start()
            self.last_used = time.monotonic()
            h, _ = self._call(dict(op="ocr", shapes=[list(a.shape[:2]) for a in arrs]),
                              b"".join(a.tobytes() for a in arrs))
            self.last_used = time.monotonic()
            self.calls += 1
        if not h.get("ok"):
            raise RuntimeError(f"vision worker: {h.get('error')}")
        return h

    def status(self) -> dict:
        return dict(running=self.running, backend=self.backend, reason=self.reason, idle_stop_s=self.idle,
                    starts=self.starts, calls=self.calls, tiles=self.tiles,
                    container=self.cfg["container"], python=self.cfg["python"])


if __name__ == "__main__":
    print(json.dumps(dict(config=config(), available=available())), file=sys.stdout)
