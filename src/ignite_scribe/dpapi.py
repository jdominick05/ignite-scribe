# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Windows DPAPI (``CryptProtectData``), per user: the history's encryption at rest. ctypes only.

``protect`` / ``unprotect`` use the CurrentUser scope, with an entropy value fixed to this application, so only
the same Windows account can decrypt a blob (another account, or the file copied to another machine, cannot), and
a DPAPI blob made by another program for this user does not decrypt here. ``CRYPTPROTECT_UI_FORBIDDEN``: never
prompt.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

if sys.platform != "win32":
    raise ImportError("ignite_scribe.dpapi needs Windows")

ENTROPY = b"ignite-scribe history v1"
CRYPTPROTECT_UI_FORBIDDEN = 0x1


class BLOB(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
crypt32.CryptProtectData.argtypes = [ctypes.POINTER(BLOB), wintypes.LPCWSTR, ctypes.POINTER(BLOB), ctypes.c_void_p,
                                     ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(BLOB)]
crypt32.CryptUnprotectData.argtypes = [ctypes.POINTER(BLOB), ctypes.c_void_p, ctypes.POINTER(BLOB), ctypes.c_void_p,
                                       ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(BLOB)]
kernel32.LocalFree.argtypes = [ctypes.c_void_p]


def _blob(data: bytes) -> tuple[BLOB, ctypes.Array]:
    buf = ctypes.create_string_buffer(data, len(data))
    return BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf


def _call(fn, data: bytes) -> bytes:
    src, keep1 = _blob(data)
    ent, keep2 = _blob(ENTROPY)
    out = BLOB()
    if fn is crypt32.CryptProtectData:
        ok = fn(ctypes.byref(src), "ignite-scribe", ctypes.byref(ent), None, None, CRYPTPROTECT_UI_FORBIDDEN,
                ctypes.byref(out))
    else:
        ok = fn(ctypes.byref(src), None, ctypes.byref(ent), None, None, CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out))
    if not ok:
        raise OSError(ctypes.get_last_error(), f"{fn.__name__} failed")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)
        del keep1, keep2


def protect(data: bytes) -> bytes:
    return _call(crypt32.CryptProtectData, data)


def unprotect(blob: bytes) -> bytes:
    return _call(crypt32.CryptUnprotectData, blob)
