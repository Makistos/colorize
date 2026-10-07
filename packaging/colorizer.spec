# PyInstaller spec for the desktop app (one-folder build).
#
#   uv sync --extra cpu --extra desktop --group build      # or --extra webgpu / cuda / directml
#   uv run pyinstaller --noconfirm packaging/colorizer.spec
#
# Output: dist/colorizer/colorizer(.exe) under the current directory. Model files are not bundled; they are read from
# ~/.cache/colorizer (see README).
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

# Gradio reads package data (frontend, templates, version files) and package metadata
# at runtime.
datas = []
for pkg in ("gradio", "gradio_client", "safehttpx", "groovy"):
    datas += collect_data_files(pkg)
for pkg in ("gradio", "gradio_client", "safehttpx", "groovy", "fastapi", "starlette",
            "pydantic", "uvicorn", "httpx", "huggingface_hub", "colorizer"):
    datas += copy_metadata(pkg)

a = Analysis(
    [os.path.join(SPECPATH, "colorizer_app.py")],
    pathex=[os.path.join(SPECPATH, "..", "src")],
    datas=datas,
    # Registered by import path in core/registry.py, so invisible to static analysis.
    hiddenimports=[
        "colorizer.models.ddcolor",
        "colorizer.models.deoldify",
        "colorizer.models.zhang",
    ],
    # Gradio inspects its own source code; keep it as .py files.
    module_collection_mode={"gradio": "py"},
    excludes=["torch", "torchvision", "diffusers", "transformers", "onnx", "onnxscript", "tkinter",
              "pytest", "mypy", "ruff"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="colorizer",
    console=sys.platform.startswith("linux"),  # log output on Linux; no console window on Windows
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="colorizer", upx=False)
