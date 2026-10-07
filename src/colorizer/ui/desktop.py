"""Desktop app: the Gradio UI in a native window (pywebview).

Install with ``uv sync --extra <runtime> --extra desktop``; run ``colorizer-app``.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from colorizer.ui.gradio_app import App

log = logging.getLogger(__name__)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


class WebviewDialogs:
    """Native file dialogs through pywebview (Qt on Linux, system dialogs elsewhere).

    Same interface as ``colorizer.ui.file_dialog``; safe to call from handler threads.
    """

    DialogUnavailable = RuntimeError

    def __init__(self, webview: Any) -> None:
        self.webview = webview

    def available(self) -> bool:
        return True

    def _window(self) -> Any:
        if not self.webview.windows:
            raise self.DialogUnavailable("the app window is closed")
        return self.webview.windows[0]

    def _kind(self, name: str) -> Any:
        enum = getattr(self.webview, "FileDialog", None)  # pywebview >= 5.1
        return getattr(enum, name) if enum else getattr(self.webview, f"{name}_DIALOG")

    def ask_save_path(self, initial: Path) -> Path | None:
        result = self._window().create_file_dialog(
            self._kind("SAVE"),
            directory=str(initial.parent),
            save_filename=initial.name,
            file_types=("Images (*.png;*.jpg;*.jpeg;*.tif;*.tiff)", "All files (*.*)"),
        )
        return _first_path(result)

    def ask_directory(self, initial: Path) -> Path | None:
        result = self._window().create_file_dialog(self._kind("FOLDER"), directory=str(initial))
        return _first_path(result)


def _first_path(result: Any) -> Path | None:
    if not result:
        return None
    return Path(result if isinstance(result, str) else result[0])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="colorizer-app", description="Colorizer desktop app")
    parser.add_argument("--port", type=int, default=0, help="local UI port (default: any free)")
    parser.add_argument("--debug", action="store_true", help="enable web inspector")
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("colorizer").setLevel(logging.INFO)
    try:
        import webview
    except ImportError:
        sys.exit("The desktop app needs pywebview: uv sync --extra cpu --extra desktop")

    app = App(dialogs=WebviewDialogs(webview))
    demo = app.build().queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port or free_port(),
        prevent_thread_lock=True,
        inbrowser=False,
        quiet=True,
        allowed_paths=[str(app.output_dir)],
    )
    try:
        webview.settings["ALLOW_DOWNLOADS"] = True  # the "Download result" button
        webview.create_window(
            "Colorizer", demo.local_url, width=1400, height=950, min_size=(900, 600)
        )
        # Qt on Linux: installable from PyPI (the GTK backend needs system Python bindings).
        webview.start(gui="qt" if sys.platform.startswith("linux") else None, debug=args.debug)
    finally:
        demo.close()
        app.worker.shutdown()


if __name__ == "__main__":
    main()
