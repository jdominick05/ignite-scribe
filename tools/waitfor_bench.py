# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What does "wait until the build finishes" cost an agent? ``wait_for`` against polling. Prints numbers only.

A fake build (``tools/fake_build.py``, synthetic text) runs in a new console window for ``--seconds``, then
prints a success line with a nonce. Three ways of noticing it are measured or computed:
- **wait_for:** one ``wait_for(text=nonce)`` MCP call on the real watcher. Measured: how late it returned after
  the line was printed, and the size of its reply.
- **text polling:** one ``get_screen_state`` per poll interval until the line shows. Its reply size is measured
  live, at each poll, and the polls needed come from the build's length.
- **screenshot polling:** one full-screen screenshot per poll. Its size is computed from the primary monitor
  with a common vision-model rule (tokens ~= w*h/750 after scaling to <= 1568 px on the long side
  and <= 1.15 MP). Other models count images differently.
Text tokens are estimated as UTF-8 bytes / 4. Each poll is also a model turn, which re-reads the context; that
cost is reported as a count, not tokens, since it depends on the agent. The reply contents are never printed
or kept: they include the user's real (redacted) windows.
"""
import argparse
import json
import math
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def image_tokens(w: int, h: int) -> int:
    s = min(1.0, 1568 / max(w, h), math.sqrt(1.15e6 / (w * h)))
    return round(int(w * s) * int(h * s) / 750)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--polls", default="2,5", help="poll intervals to compare, seconds")
    a = ap.parse_args()
    from ignite_scribe.mcp_server import Server
    from ignite_scribe.redact import Policy
    from ignite_scribe.watch import Watcher
    w = Watcher(Policy()).start()
    srv = Server(watcher=w)
    srv._watcher = lambda: w
    nonce = "n" + secrets.token_hex(4)
    stamp = Path(tempfile.gettempdir()) / f"scribe-bench-{nonce}.txt"
    build = subprocess.Popen([sys.executable, str(ROOT / "tools" / "fake_build.py"), "--seconds", str(a.seconds),
                              "--nonce", nonce, "--stamp", str(stamp)], creationflags=subprocess.CREATE_NEW_CONSOLE)
    t_start = time.time()
    state_sizes = []
    try:
        reply = srv.call("wait_for", {"text": nonce, "timeout_s": a.seconds + 30})
        t_ret = time.time()
        r = json.loads(reply)
        for _ in range(3):                                   # what one text poll costs, on this desktop
            state_sizes.append(len(srv.call("get_screen_state", {}).encode("utf-8")))
        snap = w.snapshot()[0]
    finally:
        build.kill()
        w.stop()
    t_print = float(stamp.read_text()) if stamp.is_file() else None
    stamp.unlink(missing_ok=True)
    mons = snap.monitors if snap else []
    mon = next((m for m in mons if m.left <= 0 < m.right and m.top <= 0 < m.bottom), mons[0] if mons else None)
    mw, mh = (mon.right - mon.left, mon.bottom - mon.top) if mon else (1920, 1080)
    img = image_tokens(mw, mh)
    state_tok = round(sum(state_sizes) / len(state_sizes) / 4)
    dur = (t_print or t_ret) - t_start
    out = dict(result=r["result"], build_seconds=round(dur, 1),
               wait_for=dict(calls=1, model_turns=1, reply_tokens_est=round(len(reply.encode("utf-8")) / 4),
                             late_ms=round((t_ret - t_print) * 1000) if t_print else None),
               monitor=[mw, mh], screenshot_tokens_est=img, screen_state_tokens_est=state_tok, polling={})
    for p in (float(x) for x in a.polls.split(",")):
        n = math.ceil(dur / p) + 1
        out["polling"][f"every_{p:g}s"] = dict(model_turns=n, late_ms_avg=round(p * 500),
                                               screenshot_tokens=n * img, text_state_tokens=n * state_tok)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
