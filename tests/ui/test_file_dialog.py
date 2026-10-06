import subprocess
from pathlib import Path

import pytest

from colorizer.ui import file_dialog

pytestmark = pytest.mark.ui


@pytest.fixture
def linux_desktop(monkeypatch):
    monkeypatch.setattr(file_dialog.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(file_dialog.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(file_dialog, "_has_tk", lambda: True)


def tools(cmds):
    return [Path(c[0]).name if c[0] != file_dialog.sys.executable else "tk" for c in cmds]


def test_kde_prefers_kdialog(linux_desktop, monkeypatch):
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    assert tools(file_dialog._commands(Path("/x.png"))) == ["kdialog", "zenity", "tk"]
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    assert tools(file_dialog._commands(Path("/x.png"))) == ["zenity", "kdialog", "tk"]


def test_headless_has_no_dialog(linux_desktop, monkeypatch):
    monkeypatch.delenv("DISPLAY")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert not file_dialog.available()
    with pytest.raises(file_dialog.DialogUnavailable):
        file_dialog.ask_save_path(Path("/x.png"))


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ([(0, "/home/u/a.jpg\n")], Path("/home/u/a.jpg")),
        ([(1, "")], None),  # cancelled
        ([(127, ""), (0, "/b.png\n")], Path("/b.png")),  # first tool broken, next works
    ],
)
def test_ask_save_path_results(linux_desktop, monkeypatch, results, expected):
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "KDE")
    queue = list(results)

    def fake_run(cmd, **kwargs):
        code, out = queue.pop(0)
        return subprocess.CompletedProcess(cmd, code, out, "")

    monkeypatch.setattr(file_dialog.subprocess, "run", fake_run)
    assert file_dialog.ask_save_path(Path("/start.png")) == expected


def test_all_tools_failing_is_unavailable(linux_desktop, monkeypatch):
    monkeypatch.setattr(
        file_dialog.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 3, "", "boom"),
    )
    with pytest.raises(file_dialog.DialogUnavailable):
        file_dialog.ask_save_path(Path("/start.png"))
