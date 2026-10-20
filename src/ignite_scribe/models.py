# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Download the models an installed copy needs, pinned and checked (stdlib only).

The secure-mode classifier comes from the public Hugging Face repo ``jdominick05/ignite-scribe`` at a fixed
commit. Only the files secure mode loads are fetched: it runs on the iGPU only, so the fp16 graph, the tokenizer
and the config, about 64 MB. Each file is streamed to a ``.part`` file, its size and sha256 are checked against the
values pinned here, and it is renamed into place only if both match, so a failed or tampered download never
leaves a model that secure mode would load.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "jdominick05/ignite-scribe"
REVISION = "2029176759318d2a5ebca3f940ec02604300bfa2"
SECURE_FILES = {   # published path -> (sha256, bytes)
    "scribe-guard/config.json": ("50581c7c53eb2d143bbb1ed9d9297eb357070eeadada60cba289e970db725474", 1198),
    "scribe-guard/tokenizer.json": ("f17212c38517cfedc9306b57eeffbaf58c4976fcb42e5a87b405f36a7b9a9eb2", 711551),
    "scribe-guard/model_fp16.onnx": ("facfcabd085f6b0cd4e1754ecc41786c17650e3c5a7388a3a064774778b7da6b",
                                     67025944),
}
SECURE_MB = round(sum(n for _, n in SECURE_FILES.values()) / 1e6)
EMBED_DEPS = ["numpy>=2.3", "tokenizers>=0.23", "onnxruntime-directml>=1.24"]   # the ``embed`` extra


def user_secure_dir() -> Path:
    """Where an installed copy keeps the classifier (``secure.model_dir`` looks here first)."""
    from .secure import MODEL
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ignite-scribe" / "models" / "injection" / MODEL


def url(path: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{path}"


def _fetch(path: str, dest: Path, sha: str, size: int, opener, progress) -> None:
    part = dest.with_name(dest.name + ".part")
    h, got = hashlib.sha256(), 0
    try:
        with opener(url(path)) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                got += len(chunk)
                if got > size:
                    raise ValueError(f"{path}: larger than the pinned {size} bytes")
                h.update(chunk)
                f.write(chunk)
                progress(path, got, size)
        if got != size or h.hexdigest() != sha:
            raise ValueError(f"{path}: size or sha256 does not match the pinned value; not installed")
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


def install_secure_model(dest: Path | None = None, opener=None, progress=None) -> Path:
    """Fetch and check the classifier's files into ``dest`` (default: the user's install). Files already present
    with the right sha256 are kept. Raises on any mismatch or network error, leaving no partial file."""
    dest = dest or user_secure_dir()
    dest.mkdir(parents=True, exist_ok=True)
    opener = opener or (lambda u: urllib.request.urlopen(u, timeout=60))
    progress = progress or (lambda *_: None)
    for path, (sha, size) in SECURE_FILES.items():
        target = dest / path.rsplit("/", 1)[1]
        if target.is_file() and target.stat().st_size == size and \
                hashlib.sha256(target.read_bytes()).hexdigest() == sha:
            continue
        _fetch(path, target, sha, size, opener, progress)
    return dest


def install_embed_deps() -> int:
    """pip-install the ``embed`` extra's packages into this interpreter's environment."""
    return subprocess.call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                            *EMBED_DEPS])
