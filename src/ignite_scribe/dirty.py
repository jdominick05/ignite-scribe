# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which parts of the screen changed: DXGI Desktop Duplication's dirty and move rectangles, nothing else.

``DirtyRects.poll()`` returns the desktop-coordinate rectangles that changed since the last poll, on every
attached output. Only the rectangle metadata is read: the desktop image is never mapped or copied, so no pixels
enter this process. ctypes only (raw COM vtable calls), no dependencies.

Limits:
- one duplication per output and per process; a second process can hold its own (Windows caps the number);
- the duplication is lost on a mode change, a full-screen exclusive app, or the secure desktop (UAC, lock
  screen). ``poll`` recreates it; while it cannot, ``available`` is False and the caller must rely on WinEvents;
- rotated outputs are reported unrotated (not handled yet).
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from .snapshot import Rect

if sys.platform != "win32":
    raise ImportError("ignite_scribe.dirty needs Windows")

HRESULT = ctypes.c_long
WAIT_TIMEOUT = -2005270489          # DXGI_ERROR_WAIT_TIMEOUT 0x887A0027
ACCESS_LOST = -2005270490           # DXGI_ERROR_ACCESS_LOST 0x887A0026
NOT_FOUND = -2005270526             # DXGI_ERROR_NOT_FOUND 0x887A0002
MORE_DATA = -2005270525             # DXGI_ERROR_MORE_DATA 0x887A0003


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]

    @staticmethod
    def parse(s: str) -> "GUID":
        h = s.replace("-", "")
        g = GUID(int(h[0:8], 16), int(h[8:12], 16), int(h[12:16], 16))
        for i in range(8):
            g.Data4[i] = int(h[16 + 2 * i:18 + 2 * i], 16)
        return g


IID_IDXGIFactory1 = GUID.parse("770aae78-f26f-4dba-a829-253c83d1b387")
IID_IDXGIOutput1 = GUID.parse("00cddea8-939b-4b83-a340-a685226666cc")


class OUTPUT_DESC(ctypes.Structure):
    _fields_ = [("DeviceName", wintypes.WCHAR * 32), ("DesktopCoordinates", wintypes.RECT),
                ("AttachedToDesktop", wintypes.BOOL), ("Rotation", wintypes.UINT), ("Monitor", wintypes.HMONITOR)]


class POINTER_POSITION(ctypes.Structure):
    _fields_ = [("Position", wintypes.POINT), ("Visible", wintypes.BOOL)]


class FRAME_INFO(ctypes.Structure):
    _fields_ = [("LastPresentTime", ctypes.c_longlong), ("LastMouseUpdateTime", ctypes.c_longlong),
                ("AccumulatedFrames", wintypes.UINT), ("RectsCoalesced", wintypes.BOOL),
                ("ProtectedContentMaskedOut", wintypes.BOOL), ("PointerPosition", POINTER_POSITION),
                ("TotalMetadataBufferSize", wintypes.UINT), ("PointerShapeBufferSize", wintypes.UINT)]


class MOVE_RECT(ctypes.Structure):
    _fields_ = [("SourcePoint", wintypes.POINT), ("DestinationRect", wintypes.RECT)]


def _vcall(obj, index, *args, argtypes=()):
    """Call vtable slot ``index`` of COM pointer ``obj`` (a c_void_p) -> HRESULT as int."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, *argtypes)(vtbl[index])
    return fn(obj, *args)


def _release(obj):
    if obj:
        vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vtbl[2])(obj)


# vtable slots: IUnknown 0-2, IDXGIObject 3-6, then each interface's own methods
FACTORY1_ENUMADAPTERS1 = 12
ADAPTER_ENUMOUTPUTS = 7
OUTPUT_GETDESC, OUTPUT1_DUPLICATEOUTPUT = 7, 22
DUP_ACQUIRE, DUP_DIRTY, DUP_MOVE, DUP_RELEASE = 8, 9, 10, 14

dxgi = ctypes.WinDLL("dxgi")
d3d11 = ctypes.WinDLL("d3d11")
dxgi.CreateDXGIFactory1.restype = HRESULT
dxgi.CreateDXGIFactory1.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
d3d11.D3D11CreateDevice.restype = HRESULT
d3d11.D3D11CreateDevice.argtypes = [ctypes.c_void_p, wintypes.UINT, ctypes.c_void_p, wintypes.UINT, ctypes.c_void_p,
                                    wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.POINTER(wintypes.UINT), ctypes.POINTER(ctypes.c_void_p)]


class _Output:
    def __init__(self, device, output1, origin):
        self.device, self.output1, self.origin = device, output1, origin
        self.dup = ctypes.c_void_p()
        self.error = None

    def open(self) -> bool:
        self.close_dup()
        hr = _vcall(self.output1, OUTPUT1_DUPLICATEOUTPUT, self.device, ctypes.byref(self.dup),
                    argtypes=(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)))
        self.error = None if hr == 0 else f"DuplicateOutput 0x{hr & 0xFFFFFFFF:08X}"
        return hr == 0

    def close_dup(self):
        _release(self.dup)
        self.dup = ctypes.c_void_p()

    def poll(self, buf) -> tuple[list[Rect], int]:
        """-> (changed rects in desktop coordinates, frames accumulated). Never blocks."""
        if not self.dup and not self.open():
            return [], 0
        info, res = FRAME_INFO(), ctypes.c_void_p()
        hr = _vcall(self.dup, DUP_ACQUIRE, 0, ctypes.byref(info), ctypes.byref(res),
                    argtypes=(wintypes.UINT, ctypes.POINTER(FRAME_INFO), ctypes.POINTER(ctypes.c_void_p)))
        if hr == WAIT_TIMEOUT:
            return [], 0
        if hr != 0:
            self.error = f"AcquireNextFrame 0x{hr & 0xFFFFFFFF:08X}" + (" (access lost)" if hr == ACCESS_LOST else "")
            self.close_dup()
            return [], 0
        rects = []
        try:
            _release(res)                       # the desktop image itself: never mapped, never read
            if info.AccumulatedFrames and info.TotalMetadataBufferSize:
                ox, oy = self.origin
                need = wintypes.UINT()
                if _vcall(self.dup, DUP_MOVE, len(buf), buf, ctypes.byref(need),
                          argtypes=(wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(wintypes.UINT))) == 0:
                    for m in (MOVE_RECT * (need.value // ctypes.sizeof(MOVE_RECT))).from_buffer(buf):
                        r = m.DestinationRect
                        rects.append(Rect(r.left + ox, r.top + oy, r.right + ox, r.bottom + oy))
                if _vcall(self.dup, DUP_DIRTY, len(buf), buf, ctypes.byref(need),
                          argtypes=(wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(wintypes.UINT))) == 0:
                    for r in (wintypes.RECT * (need.value // ctypes.sizeof(wintypes.RECT))).from_buffer(buf):
                        rects.append(Rect(r.left + ox, r.top + oy, r.right + ox, r.bottom + oy))
                else:                           # buffer too small: report the whole output as changed
                    rects.append(Rect(ox, oy, ox + (1 << 15), oy + (1 << 15)))
        finally:
            _vcall(self.dup, DUP_RELEASE)
        return rects, info.AccumulatedFrames

    def close(self):
        self.close_dup()
        _release(self.output1)
        _release(self.device)


class DirtyRects:
    """Desktop Duplication on every attached output of every adapter."""

    def __init__(self):
        self.outputs: list[_Output] = []
        self.error = None
        self.buf = (ctypes.c_ubyte * (1 << 16))()
        factory = ctypes.c_void_p()
        hr = dxgi.CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(factory))
        if hr != 0:
            self.error = f"CreateDXGIFactory1 0x{hr & 0xFFFFFFFF:08X}"
            return
        try:
            a = 0
            while True:
                adapter = ctypes.c_void_p()
                if _vcall(factory, FACTORY1_ENUMADAPTERS1, a, ctypes.byref(adapter),
                          argtypes=(wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))) != 0:
                    break
                try:
                    self._adapter(adapter)
                finally:
                    _release(adapter)
                a += 1
        finally:
            _release(factory)
        if not self.outputs and not self.error:
            self.error = "no attached output could be duplicated"

    def _adapter(self, adapter):
        o = 0
        while True:
            output = ctypes.c_void_p()
            if _vcall(adapter, ADAPTER_ENUMOUTPUTS, o, ctypes.byref(output),
                      argtypes=(wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))) != 0:
                return
            o += 1
            try:
                desc = OUTPUT_DESC()
                _vcall(output, OUTPUT_GETDESC, ctypes.byref(desc), argtypes=(ctypes.POINTER(OUTPUT_DESC),))
                if not desc.AttachedToDesktop:
                    continue
                output1 = ctypes.c_void_p()
                if _vcall(output, 0, ctypes.byref(IID_IDXGIOutput1), ctypes.byref(output1),
                          argtypes=(ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))) != 0:
                    continue
                device, level, ctx = ctypes.c_void_p(), wintypes.UINT(), ctypes.c_void_p()
                hr = d3d11.D3D11CreateDevice(adapter, 0, None, 0, None, 0, 7, ctypes.byref(device),
                                             ctypes.byref(level), ctypes.byref(ctx))
                _release(ctx)
                if hr != 0:
                    _release(output1)
                    self.error = f"D3D11CreateDevice 0x{hr & 0xFFFFFFFF:08X}"
                    continue
                out = _Output(device, output1, (desc.DesktopCoordinates.left, desc.DesktopCoordinates.top))
                if not out.open():
                    self.error = out.error
                self.outputs.append(out)
            finally:
                _release(output)

    @property
    def available(self) -> bool:
        return any(o.dup for o in self.outputs)

    def poll(self) -> list[Rect]:
        rects = []
        for o in self.outputs:
            rects += o.poll(self.buf)[0]
        errs = [o.error for o in self.outputs if o.error]
        self.error = errs[0] if errs else None
        return rects

    def close(self):
        for o in self.outputs:
            o.close()
        self.outputs = []
