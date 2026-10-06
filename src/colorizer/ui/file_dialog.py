"""Native "Save as" dialog for the local UI.

The UI server runs on the user's machine, so it can open a desktop dialog there. Each
dialog runs in a subprocess so no GUI toolkit state lives in the server process.
Preference: kdialog on KDE, zenity elsewhere on Linux, tkinter on any platform.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

TITLE = "Save colorized image"
_PATTERNS = "*.png *.jpg *.jpeg *.tif *.tiff"

# Exits 0 with the path (empty if cancelled) or 3 on error, so failures aren't mistaken
# for a cancel (which kdialog/zenity report as exit 1).
_TK_SCRIPT = """
import sys
from pathlib import Path
try:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    initial = Path(sys.argv[1])
    path = filedialog.asksaveasfilename(
        title=sys.argv[2], initialdir=str(initial.parent), initialfile=initial.name,
        filetypes=[("Images", sys.argv[3]), ("All files", "*")],
    )
except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(3)
print(path or "")
"""


class DialogUnavailable(RuntimeError):
    pass


def _has_tk() -> bool:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return False
    return True


def _commands(initial: Path) -> list[list[str]]:
    cmds: list[list[str]] = []
    if sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return []  # headless: no dialog can be shown
        kde = "KDE" in os.environ.get("XDG_CURRENT_DESKTOP", "").upper()
        kdialog = [
            "kdialog",
            "--title",
            TITLE,
            "--getsavefilename",
            str(initial),
            f"Images ({_PATTERNS})",
        ]
        # zenity >= 4 asks before overwriting by default.
        zenity = [
            "zenity",
            "--file-selection",
            "--save",
            f"--title={TITLE}",
            f"--filename={initial}",
            f"--file-filter=Images | {_PATTERNS}",
        ]
        for name, cmd in (
            (("kdialog", kdialog), ("zenity", zenity))
            if kde
            else (("zenity", zenity), ("kdialog", kdialog))
        ):
            if shutil.which(name):
                cmds.append(cmd)
    if _has_tk():
        cmds.append([sys.executable, "-c", _TK_SCRIPT, str(initial), TITLE, _PATTERNS])
    return cmds


def available() -> bool:
    return bool(_commands(Path.home()))


def ask_save_path(initial: Path) -> Path | None:
    """Show a native save dialog. Returns the chosen path, or ``None`` if cancelled.

    Raises ``DialogUnavailable`` if no dialog could be shown.
    """
    for cmd in _commands(initial):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError as e:
            log.debug("%s failed: %s", cmd[0], e)
            continue
        # kdialog and zenity exit 1 on cancel; anything else is a failure to show.
        if result.returncode == 0:
            chosen = result.stdout.strip()
            return Path(chosen) if chosen else None
        if result.returncode == 1:
            return None
        log.debug("%s exited %d: %s", cmd[0], result.returncode, result.stderr.strip())
    raise DialogUnavailable("no file dialog available (kdialog, zenity or tkinter)")
