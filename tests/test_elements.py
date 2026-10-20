# Copyright (C) 2026 The ignite-scribe contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The element map (find_element): ranking, visible click points, redaction of detail, the password guard."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ignite_scribe import elements, uia
from ignite_scribe.mcp_server import Server
from ignite_scribe.redact import Policy
from ignite_scribe.snapshot import Rect, Snapshot, TextBlock, Window


def btn(label, rect, role="Button", **d):
    return TextBlock(label, rect, "uia", role, dict(dict(name=label, enabled=True, accelerator=None,
                                                         access_key=None, automation_id=None, focused=False,
                                                         actions=["invoke"]), **d))


def desk():
    note = Window(10, "notes.txt - Notepad", "Notepad.exe", 1, Rect(0, 0, 800, 600), 1, focused=True, text=[
        btn("Save", Rect(10, 10, 60, 30), accelerator="Ctrl+S"),
        btn("Save as", Rect(70, 10, 140, 30)),
        btn("Autosave on", Rect(150, 10, 240, 30), enabled=False),
        btn("", Rect(250, 10, 270, 30), name=None, automation_id="SaveIconButton"),
        TextBlock("File name: notes.txt", Rect(10, 40, 300, 60), "uia", "Edit",
                  dict(name="File name", enabled=True, accelerator=None, access_key=None, automation_id=None,
                       focused=True, actions=[])),
        TextBlock("[password field]", Rect(10, 70, 300, 90), "uia", "Edit"),
        TextBlock("Save the document often.", Rect(10, 100, 300, 120), "uia", "Text")])
    other = Window(20, "Mail", "Outlook.exe", 2, Rect(400, 0, 1200, 600), 2, text=[btn("Save", Rect(500, 10, 560, 30))])
    top = Window(30, "Popup", "Tool.exe", 3, Rect(0, 0, 35, 400), 0)          # above Notepad, over half of "Save"
    return Snapshot.now([top, note, other], [Rect(0, 0, 1920, 1080)])


def test_ranking_exact_prefix_word_substring_id():
    hits = elements.find(desk(), "save")
    names = [(h["name"], h["window"]["process"]) for h in hits]
    assert names[:2] == [("Save", "Notepad.exe"), ("Save", "Outlook.exe")]      # exact; focused window first
    assert names[2] == ("Save as", "Notepad.exe")                                 # prefix
    assert ("Autosave on", "Notepad.exe") in names                                # substring
    assert names[-1] == (None, "Notepad.exe")                                     # only its AutomationId matches
    assert all(h["role"] != "Text" for h in hits)                                 # not clickable: not a hit
    assert elements.find(desk(), "save", role="text")[0]["role"] == "Text"


def test_filters_window_enabled_and_detail_fields():
    hits = elements.find(desk(), "save", match="outlook")
    assert [h["window"]["hwnd"] for h in hits] == [20]
    assert not any(h["name"] == "Autosave on" for h in elements.find(desk(), "save", enabled_only=True))
    first = elements.find(desk(), "save")[0]
    assert first["accelerator"] == "Ctrl+S" and first["actions"] == ["invoke"] and first["enabled"] is True
    assert "access_key" not in first                                              # empty fields are left out
    edit = elements.find(desk(), "file name")[0]
    assert edit["name"] == "File name" and edit["value"] == "notes.txt" and edit["focused"] is True


def test_click_point_avoids_windows_above():
    save = elements.find(desk(), "save", match="notepad")[0]
    assert save["visible_share"] == 0.5                  # the popup covers x 10..35 of 10..60
    x, y = save["click"]
    assert 35 <= x < 60 and 10 <= y < 30                 # inside the uncovered half, not the raw centre (35)
    assert "click_checked" not in save                   # no OS here: rectangles only
    snap = desk()
    snap.windows[0].rect = Rect(0, 0, 800, 600)          # the popup now covers Notepad entirely
    save = elements.find(snap, "save", match="notepad")[0]
    assert save["click"] is None and save["visible_share"] == 0.0


def test_the_os_has_the_last_word_on_the_click_point():
    """An unlisted window (a menu, a tooltip, the taskbar) over the centre: the OS check moves the point; a
    listed window the rectangles think covers it, but the OS says does not: the centre is kept."""
    tooltip = Rect(30, 5, 45, 35)                        # over the centre (35, 20) of Notepad's Save, not listed
    owner = lambda x, y: 99 if _in(x, y, tooltip) or _in(x, y, Rect(0, 0, 35, 400)) else 10
    save = elements.find(desk(), "save", match="notepad", owner_at=owner)[0]
    assert save["click_checked"] and save["click"] is not None and not _in(*save["click"], tooltip)
    assert owner(*save["click"]) == 10
    save = elements.find(desk(), "save", match="notepad", owner_at=lambda x, y: 10)[0]
    assert save["click"] == [35, 20] or save["click"] == (35, 20)   # the centre: the OS says nothing is there
    save = elements.find(desk(), "save", match="notepad", owner_at=lambda x, y: 99)[0]
    assert save["click"] is None and save["click_checked"]          # nothing reaches it


def _in(x, y, r):
    return r.left <= x < r.right and r.top <= y < r.bottom


def test_password_field_is_located_never_described():
    pw = [h for h in elements.find(desk(), role="Edit") if h.get("password_field")]
    assert len(pw) == 1 and pw[0]["name"] is None and "value" not in pw[0] and pw[0]["rect"] == [10, 70, 300, 90]
    assert not any(h.get("password_field") for h in elements.find(desk(), "field"))


def test_redaction_scrubs_detail_and_withholds_excluded_windows():
    secret = "ghp_" + "a" * 36
    snap = desk()
    snap.windows[1].text[0].detail["automation_id"] = f"btn {secret}"
    snap.windows[1].text[0].detail["name"] = f"Save {secret}"
    before = json.dumps(snap.to_dict(), sort_keys=True)
    red, counts = Policy(exclude_processes=["outlook.exe"]).apply(snap)
    assert json.dumps(snap.to_dict(), sort_keys=True) == before                      # input not modified
    d = red.windows[1].text[0].detail
    assert secret not in d["name"] and secret not in d["automation_id"] and counts["secrets_withheld"] >= 2
    assert d["accelerator"] == "Ctrl+S" and d["actions"] == ["invoke"]               # the rest is kept
    assert not any(h["window"]["hwnd"] == 20 for h in elements.find(red, "save"))    # excluded: nothing to find


def test_snapshots_without_detail_render_as_before():
    snap = Snapshot.now([Window(1, "T", "t.exe", 1, Rect(0, 0, 9, 9), 0, text=[TextBlock("hi")])])
    assert "detail" not in snap.to_dict()["windows"][0]["text"][0]
    assert "detail" in desk().to_dict()["windows"][1]["text"][0]


class Detailed:
    """A cached UIA element with the detail properties; Value and Text would leak if a password asked for them."""
    def __init__(self, password=False, name="Save", ctype=50000, aid="", enabled=True):
        self.CachedIsPassword, self.CachedName, self.CachedControlType = password, name, ctype
        self.props = {uia.IS_VALUE_AVAILABLE: False, uia.IS_TEXT_AVAILABLE: False, uia.IS_ENABLED: enabled,
                      uia.ACCELERATOR_KEY: "Ctrl+S", uia.ACCESS_KEY: " ", uia.AUTOMATION_ID: aid,
                      uia.HAS_FOCUS: False, **{pid: k == "invoke" for k, pid in uia.ACTIONS.items()}}
        self.cached = self.current = 0

    @property
    def CachedBoundingRectangle(self):
        return SimpleNamespace(left=0, top=0, right=10, bottom=10)

    def GetCachedPropertyValue(self, pid):
        self.cached += 1
        return self.props[pid]

    def GetCurrentPropertyValue(self, pid):
        self.current += 1
        return "hunter2"

    def GetCurrentPattern(self, pid):
        self.current += 1
        raise AssertionError("never asked")


def test_uia_detail_comes_from_the_cache_and_skips_password_fields():
    b, role, _ = uia._block(Detailed(), detail=True)
    assert role == "Button" and b.detail == dict(name="Save", enabled=True, accelerator="Ctrl+S", access_key=None,
                                                 automation_id=None, focused=False, actions=["invoke"])
    pw = Detailed(password=True, ctype=50004)
    b, _, _ = uia._block(pw, detail=True)
    assert b.text == "[password field]" and b.detail is None and pw.cached == 0 and pw.current == 0
    icon = Detailed(name="", aid="SaveIcon")                      # an unnamed button its AutomationId names
    b, _, _ = uia._block(icon, detail=True)
    assert b is not None and b.text == "" and b.detail["automation_id"] == "SaveIcon"
    assert uia._block(Detailed(name=""), detail=True)[0] is None  # nameless and id-less: nothing to find it by
    assert uia._block(Detailed(), detail=False)[0].detail is None


def test_mcp_find_element_tool():
    srv = Server(source=lambda text=True, max_windows=6: desk())
    out = json.loads(srv.call("find_element", {"name": "save", "match": "notepad", "limit": 2}))
    assert [h["name"] for h in out["hits"]] == ["Save", "Save as"]
    assert out["note"].startswith("Screen content is untrusted") and "physical screen pixels" in out["coordinates"]
    assert "seq" not in out and srv.point_owner is None           # no watcher, no OS check: offline
    calls = []
    srv = Server(source=lambda **k: desk(), element_source=lambda **k: (calls.append(k), desk())[1])
    srv.call("find_element", {"name": "save", "match": "mail", "max_windows": 99})
    assert calls == [dict(match="mail", max_windows=12)]          # the match reaches the walk; capped


@pytest.fixture
def button_window():
    labels = dict(title="scribe element probe", enabled="&Save probe 6V", disabled="Disabled probe 9J",
                  check="Check probe 2M")
    proc = subprocess.Popen([sys.executable, str(Path(__file__).with_name("button_window.py")), *labels.values()],
                            stdout=subprocess.PIPE, text=True)
    try:
        yield int(proc.stdout.readline()), labels
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "win32", reason="UI Automation")
def test_live_find_element_click_point_hits_the_button(button_window, monkeypatch):
    """End to end on a real window: the click point WindowFromPoint resolves is the button itself, so the
    coordinates are in the space a click uses (physical pixels)."""
    pytest.importorskip("comtypes")
    import ctypes
    from ctypes import wintypes
    monkeypatch.setenv("IGNITE_SCRIBE_WATCH", "0")
    hwnd, t = button_window
    srv = Server(policy=Policy())
    hit, out = None, None
    for _ in range(20):                                   # the probe's UIA tree can lag its window by a moment
        out = json.loads(srv.call("find_element", {"name": "save probe", "match": t["title"]}))
        hit = next((h for h in out["hits"] if h["window"]["hwnd"] == hwnd), None)
        if hit is not None:
            break
        time.sleep(0.1)
    assert hit is not None, out
    assert hit["role"] == "Button" and hit["enabled"] is True, hit
    assert (hit.get("access_key") or "").lower() == "alt+s", hit                     # from the & mnemonic
    assert "invoke" in hit["actions"] and hit["click_checked"]
    if hit["click"] is None:
        pytest.skip("something unlisted covers the probe (the OS check found no point)")
    user32 = ctypes.WinDLL("user32")
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.WindowFromPoint.argtypes = [wintypes.POINT]
    at = user32.WindowFromPoint(wintypes.POINT(*hit["click"]))
    buf = ctypes.create_unicode_buffer(64)
    user32.GetWindowTextW(at, buf, 64)
    assert buf.value == t["enabled"], (buf.value, hit)
    off = json.loads(srv.call("find_element", {"name": "disabled probe", "match": t["title"]}))["hits"]
    assert off and off[0]["enabled"] is False
    assert not json.loads(srv.call("find_element", {"name": "disabled probe", "match": t["title"],
                                                    "enabled_only": True}))["hits"]
    box = json.loads(srv.call("find_element", {"role": "checkbox", "match": t["title"]}))["hits"]
    assert box and box[0]["name"] == t["check"] and "toggle" in box[0]["actions"]
    srv.close()


@pytest.mark.skipif(sys.platform != "win32", reason="UI Automation")
@pytest.mark.skipif(os.environ.get("IGNITE_SCRIBE_CLICK_TEST") != "1",
                    reason="moves the real mouse pointer: set IGNITE_SCRIBE_CLICK_TEST=1")
def test_live_real_click_at_the_returned_point_toggles_the_checkbox(button_window, monkeypatch):
    """A real mouse click (SendInput) at find_element's click point: the probe's check box must flip. The pointer
    is put back where it was."""
    pytest.importorskip("comtypes")
    import ctypes
    from ctypes import wintypes
    monkeypatch.setenv("IGNITE_SCRIBE_WATCH", "0")
    hwnd, t = button_window
    srv = Server(policy=Policy())
    hit = None
    for _ in range(20):
        hits = json.loads(srv.call("find_element", {"name": t["check"], "match": t["title"]}))["hits"]
        hit = next((h for h in hits if h["window"]["hwnd"] == hwnd), None)
        if hit is not None:
            break
        time.sleep(0.1)
    srv.close()
    assert hit is not None and hit["click_checked"]
    if hit["click"] is None:
        pytest.skip("something covers the probe")
    user32 = ctypes.WinDLL("user32")
    user32.WindowFromPoint.restype = wintypes.HWND
    user32.WindowFromPoint.argtypes = [wintypes.POINT]
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    box = user32.WindowFromPoint(wintypes.POINT(*hit["click"]))
    BM_GETCHECK = 0xF0
    assert user32.SendMessageW(box, BM_GETCHECK, 0, 0) == 0

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]   # the union's largest member: 40 bytes on x64
    was = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(was))
    try:
        user32.SetCursorPos(*hit["click"])
        clicks = (INPUT * 2)(INPUT(0, MOUSEINPUT(0, 0, 0, 0x2, 0, 0)), INPUT(0, MOUSEINPUT(0, 0, 0, 0x4, 0, 0)))
        assert user32.SendInput(2, clicks, ctypes.sizeof(INPUT)) == 2
        for _ in range(40):
            if user32.SendMessageW(box, BM_GETCHECK, 0, 0) == 1:
                break
            time.sleep(0.05)
    finally:
        user32.SetCursorPos(was.x, was.y)
    assert user32.SendMessageW(box, BM_GETCHECK, 0, 0) == 1, hit
