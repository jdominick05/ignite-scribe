# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AES-256-GCM through Windows CNG (``bcrypt.dll``), ctypes only: the history's per-chunk cipher.

The history keeps one random 256-bit data key, wrapped with DPAPI for the current user (``dpapi``), and seals
each chunk with it: a fresh 96-bit nonce per chunk, a 128-bit tag, and the chunk's clear metadata as associated
data, so a row's time range or kind cannot be swapped without the tag failing. DPAPI per chunk would cost about
0.13 ms of fixed overhead per call; CNG's AES uses AES-NI.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

if sys.platform != "win32":
    raise ImportError("ignite_scribe.aesgcm needs Windows")

NONCE, TAG = 12, 16
STATUS_AUTH_TAG_MISMATCH = 0xC000A002

bcrypt = ctypes.WinDLL("bcrypt")
HANDLE = ctypes.c_void_p
NTSTATUS = ctypes.c_long


class AuthInfo(ctypes.Structure):          # BCRYPT_AUTHENTICATED_CIPHER_MODE_INFO
    _fields_ = [("cbSize", wintypes.ULONG), ("dwInfoVersion", wintypes.ULONG),
                ("pbNonce", ctypes.c_void_p), ("cbNonce", wintypes.ULONG),
                ("pbAuthData", ctypes.c_void_p), ("cbAuthData", wintypes.ULONG),
                ("pbTag", ctypes.c_void_p), ("cbTag", wintypes.ULONG),
                ("pbMacContext", ctypes.c_void_p), ("cbMacContext", wintypes.ULONG),
                ("cbAAD", wintypes.ULONG), ("cbData", ctypes.c_ulonglong), ("dwFlags", wintypes.ULONG)]


for name, args in (("BCryptOpenAlgorithmProvider", [ctypes.POINTER(HANDLE), wintypes.LPCWSTR, wintypes.LPCWSTR,
                                                     wintypes.ULONG]),
                   ("BCryptSetProperty", [HANDLE, wintypes.LPCWSTR, ctypes.c_void_p, wintypes.ULONG, wintypes.ULONG]),
                   ("BCryptGenerateSymmetricKey", [HANDLE, ctypes.POINTER(HANDLE), ctypes.c_void_p, wintypes.ULONG,
                                                   ctypes.c_char_p, wintypes.ULONG, wintypes.ULONG]),
                   ("BCryptEncrypt", [HANDLE, ctypes.c_char_p, wintypes.ULONG, ctypes.c_void_p, ctypes.c_void_p,
                                      wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
                                      wintypes.ULONG]),
                   ("BCryptDecrypt", [HANDLE, ctypes.c_char_p, wintypes.ULONG, ctypes.c_void_p, ctypes.c_void_p,
                                      wintypes.ULONG, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
                                      wintypes.ULONG]),
                   ("BCryptDestroyKey", [HANDLE]),
                   ("BCryptCloseAlgorithmProvider", [HANDLE, wintypes.ULONG])):
    f = getattr(bcrypt, name)
    f.argtypes, f.restype = args, NTSTATUS


def _check(status, what):
    if status != 0:
        raise OSError(f"{what} failed: NTSTATUS 0x{status & 0xFFFFFFFF:08X}")


class AesGcm:
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("AES-256 needs a 32-byte key")
        self._alg, self._key = HANDLE(), HANDLE()
        _check(bcrypt.BCryptOpenAlgorithmProvider(ctypes.byref(self._alg), "AES", None, 0), "open AES")
        mode = ctypes.create_unicode_buffer("ChainingModeGCM")
        _check(bcrypt.BCryptSetProperty(self._alg, "ChainingMode", mode, ctypes.sizeof(mode), 0), "set GCM")
        _check(bcrypt.BCryptGenerateSymmetricKey(self._alg, ctypes.byref(self._key), None, 0, key, len(key), 0),
               "import key")

    def _info(self, nonce, aad, tag):
        i = AuthInfo()
        i.cbSize, i.dwInfoVersion = ctypes.sizeof(AuthInfo), 1
        i.pbNonce, i.cbNonce = ctypes.cast(nonce, ctypes.c_void_p), NONCE
        i.pbAuthData, i.cbAuthData = ctypes.cast(aad, ctypes.c_void_p), len(aad) if aad else 0
        i.pbTag, i.cbTag = ctypes.cast(tag, ctypes.c_void_p), TAG
        return i

    def seal(self, plain: bytes, aad: bytes = b"") -> bytes:
        """-> nonce | ciphertext | tag."""
        nonce = ctypes.create_string_buffer(os.urandom(NONCE), NONCE)
        a = ctypes.create_string_buffer(aad, len(aad)) if aad else None
        tag = ctypes.create_string_buffer(TAG)
        out = ctypes.create_string_buffer(max(1, len(plain)))
        n = wintypes.ULONG()
        info = self._info(nonce, a, tag)
        _check(bcrypt.BCryptEncrypt(self._key, plain, len(plain), ctypes.byref(info), None, 0, out, len(plain),
                                    ctypes.byref(n), 0), "encrypt")
        return nonce.raw + out.raw[:n.value] + tag.raw

    def open(self, blob: bytes, aad: bytes = b"") -> bytes:
        if len(blob) < NONCE + TAG:
            raise ValueError("sealed blob too short")
        nonce = ctypes.create_string_buffer(blob[:NONCE], NONCE)
        body = blob[NONCE:-TAG]
        tag = ctypes.create_string_buffer(blob[-TAG:], TAG)
        a = ctypes.create_string_buffer(aad, len(aad)) if aad else None
        out = ctypes.create_string_buffer(max(1, len(body)))
        n = wintypes.ULONG()
        info = self._info(nonce, a, tag)
        st = bcrypt.BCryptDecrypt(self._key, body, len(body), ctypes.byref(info), None, 0, out, len(body),
                                  ctypes.byref(n), 0)
        if st & 0xFFFFFFFF == STATUS_AUTH_TAG_MISMATCH:
            raise ValueError("history chunk failed authentication (tampered, or another key)")
        _check(st, "decrypt")
        return out.raw[:n.value]

    def close(self):
        if self._key:
            bcrypt.BCryptDestroyKey(self._key)
            self._key = HANDLE()
        if self._alg:
            bcrypt.BCryptCloseAlgorithmProvider(self._alg, 0)
            self._alg = HANDLE()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
