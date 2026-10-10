import json

import gradio as gr
import numpy as np
import pytest
from gradio_client import Client, handle_file
from PIL import Image

from colorizer.core.params import Param
from colorizer.core.postprocess import POSTPROCESS_PARAMS
from colorizer.core.registry import Registry
from colorizer.ui.gradio_app import VIEWS, App, widget_for, widget_value
from tests.conftest import DummyModel

pytestmark = pytest.mark.ui


@pytest.fixture
def app(registry, tmp_path):
    a = App(registry=registry, output_dir=tmp_path / "out", config_dir=tmp_path / "cfg")
    a.last_save_dir = tmp_path / "saved"
    a.last_save_dir.mkdir()
    yield a
    a.worker.shutdown()


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "photo.png"
    Image.fromarray(np.tile(np.arange(0, 250, 5, dtype=np.uint8), (40, 1))).save(p)
    return str(p)


def defaults(app, restore=()):
    """Flat widget values: model slots, restore chain + its slots, postprocess."""
    return [
        *(s.param.default for s in app.slots),
        list(restore),
        *(s.param.default for s in app.rslots),
        *(p.default for p in POSTPROCESS_PARAMS),
    ]


def run_colorize(app, image_path, values, model="dummy", view=VIEWS[0]):
    *_, last = app.colorize(image_path, model, "cpu", "png", 95, view, *values)
    return last


@pytest.mark.parametrize(
    ("param", "widget"),
    [
        (Param("n", "int", 512, min=256, max=1024, step=64), gr.Slider),
        (Param("n", "int", 256, min=256, max=256), gr.Number),
        (Param("x", "float", 1.0), gr.Number),
        (Param("s", "seed", 0), gr.Number),
        (Param("b", "bool", True), gr.Checkbox),
        (Param("c", "choice", "a", choices=("a", "b")), gr.Dropdown),
        (Param("t", "str", ""), gr.Textbox),
        (Param("p", "points", []), gr.Textbox),
    ],
)
def test_widget_for_each_kind(param, widget):
    with gr.Blocks():
        assert isinstance(widget_for(param), widget)


def test_widget_value_conversion():
    assert widget_value(Param("n", "int", 1), 512.0) == 512
    pts = Param("p", "points", [])
    assert widget_value(pts, "") == []
    assert widget_value(pts, json.dumps([[1, 2, [3, 4, 5]]])) == [(1, 2, (3, 4, 5))]


def test_every_registered_model_gets_widgets():
    app = App(registry=Registry())  # real built-ins
    for model_id, cls in app.models.items():
        assert [s.param for s in app.slots if s.model_id == model_id] == list(cls.params)
    app.build()
    app.worker.shutdown()


def test_colorize_then_postprocess_without_reinference(app, image_path):
    values = defaults(app)
    (before, after), out, status = run_colorize(app, image_path, values)
    assert before.shape == after.shape == (40, 50, 3)
    assert np.ptp(before.astype(int), axis=2).max() <= 1  # gray
    assert np.ptp(after.astype(int), axis=2).max() > 5  # coloured
    assert out.endswith("photo_colorized.png") and status.startswith("Done")
    assert app.session.inference_count == 1

    gray_values = [*values[: -len(POSTPROCESS_PARAMS)], 0.0, 0.0, 0.0, 1.0, 0.0]  # saturation 0
    (_, after2), _, status2 = app.rerender(image_path, "dummy", "png", 95, VIEWS[0], *gray_values)
    assert np.ptp(after2.astype(int), axis=2).max() <= 1
    assert status2.startswith("Updated")
    assert app.session.inference_count == 1

    # Same model params again: Colorize hits the cache too.
    run_colorize(app, image_path, values)
    assert app.session.inference_count == 1


def test_rerender_with_changed_model_params_does_not_infer(app, image_path):
    values = defaults(app)
    run_colorize(app, image_path, values)
    changed = list(values)
    changed[0] = 64  # dummy "size"
    _, _, status = app.rerender(image_path, "dummy", "png", 95, VIEWS[0], *changed)
    assert "Colorize" in status
    assert app.session.inference_count == 1


def test_rerender_format_change_writes_new_file(app, image_path):
    run_colorize(app, image_path, defaults(app))
    _, out, _ = app.rerender(image_path, "dummy", "jpg", 80, VIEWS[0], *defaults(app))
    assert out.endswith(".jpg") and Image.open(out).format == "JPEG"


def test_invalid_value_raises_gradio_error(app, image_path):
    values = defaults(app)
    values[0] = 17
    with pytest.raises(gr.Error):
        run_colorize(app, image_path, values)
    assert DummyModel.instances == []


def test_model_load_error_surfaces(app, image_path, monkeypatch):
    def broken(self, device):
        raise FileNotFoundError("weights missing; run the export")

    monkeypatch.setattr(DummyModel, "load", broken)
    with pytest.raises(gr.Error, match="weights missing"):
        run_colorize(app, image_path, defaults(app))


def test_cancel_stops_job(app, image_path, monkeypatch):
    import threading

    started, release = threading.Event(), threading.Event()
    original = DummyModel.load

    def slow_load(self, device):
        started.set()
        release.wait(5)
        original(self, device)

    monkeypatch.setattr(DummyModel, "load", slow_load)
    gen = app.colorize(image_path, "dummy", "cpu", "png", 95, VIEWS[0], *defaults(app))
    next(gen)  # first progress update
    started.wait(5)
    app.cancel()
    release.set()
    assert list(gen)[-1][2] == "Cancelled."
    assert app.session.inference_count == 0


def test_end_to_end_via_gradio_client(app, image_path):
    demo = app.build().queue()
    _, url, _ = demo.launch(
        prevent_thread_lock=True,
        server_name="127.0.0.1",
        server_port=None,
        quiet=True,
        allowed_paths=[str(app.output_dir)],
    )
    try:
        client = Client(url, verbose=False)
        img = handle_file(image_path)
        result = client.predict(
            img, "dummy", "cpu", "png", 95, VIEWS[0], *defaults(app), api_name="/colorize"
        )
        assert len(result) == 3 and result[2].startswith("Done")
        result = client.predict(
            img, "dummy", "png", 95, VIEWS[0], *defaults(app), api_name="/rerender"
        )
        assert result[2].startswith("Updated")
        assert app.session.inference_count == 1
    finally:
        demo.close()


# --- saving to a chosen path -----------------------------------------------------------


def save(app, image_path, dest, overwrite=False, fmt="png"):
    return app.save_to(image_path, "dummy", fmt, 90, dest, overwrite, *defaults(app))


def test_suggested_path_and_extension_follow_format(app, image_path):
    suggested = app.suggest_save_path(image_path, "jpg")
    assert suggested == str(app.last_save_dir / "photo_colorized.jpg")
    assert app.retarget_extension(suggested, "tiff").endswith("photo_colorized.tiff")


def test_save_requires_a_result(app, image_path):
    _, status = save(app, image_path, str(app.last_save_dir / "x.png"))
    assert "Colorize" in status
    assert not (app.last_save_dir / "x.png").exists()


def test_save_uses_extension_for_format_without_reinference(app, image_path, tmp_path):
    run_colorize(app, image_path, defaults(app))
    dest = tmp_path / "elsewhere" / "kuva2.jpg"
    path, status = save(app, image_path, str(dest), fmt="png")
    assert path == str(dest) and status.startswith("Saved")
    assert Image.open(dest).format == "JPEG"
    assert app.session.inference_count == 1


def test_save_remembers_folder_across_sessions(app, registry, image_path, tmp_path):
    run_colorize(app, image_path, defaults(app))
    dest = tmp_path / "album" / "a.png"
    save(app, image_path, str(dest))
    again = App(registry=registry, output_dir=tmp_path / "o2", config_dir=tmp_path / "cfg")
    assert again.last_save_dir == dest.parent
    again.worker.shutdown()


def test_existing_file_needs_overwrite(app, image_path):
    run_colorize(app, image_path, defaults(app))
    dest = app.last_save_dir / "keep.png"
    dest.write_bytes(b"original")
    _, status = save(app, image_path, str(dest))
    assert "already exists" in status and dest.read_bytes() == b"original"
    _, status = save(app, image_path, str(dest), overwrite=True)
    assert status.startswith("Saved") and Image.open(dest).format == "PNG"


@pytest.mark.parametrize(
    ("dest", "expected"),
    [
        ("{dir}", "photo_colorized.png"),  # a folder gets the default name
        ("noext", "noext.png"),  # missing extension comes from Format
        ("relative/b.tif", "relative/b.tif"),  # relative to the last folder
    ],
)
def test_save_path_resolution(app, image_path, dest, expected):
    run_colorize(app, image_path, defaults(app))
    base = app.last_save_dir  # saving moves last_save_dir to the new file's folder
    path, status = save(app, image_path, dest.format(dir=base))
    assert status.startswith("Saved")
    assert path == str(base / expected)


def test_save_as_uses_dialog_choice(app, image_path, tmp_path, monkeypatch):
    run_colorize(app, image_path, defaults(app))
    chosen = tmp_path / "picked.jpg"
    chosen.write_bytes(b"old")  # dialog confirmed the overwrite
    monkeypatch.setattr("colorizer.ui.file_dialog.ask_save_path", lambda initial: chosen)
    path, status = app.save_as(image_path, "dummy", "png", 90, "", False, *defaults(app))
    assert path == str(chosen) and status.startswith("Saved")
    assert Image.open(chosen).format == "JPEG"


def test_save_as_cancel_and_unavailable(app, image_path, monkeypatch):
    from colorizer.ui import file_dialog

    run_colorize(app, image_path, defaults(app))
    monkeypatch.setattr(file_dialog, "ask_save_path", lambda initial: None)
    assert app.save_as(image_path, "dummy", "png", 90, "", False, *defaults(app))[1] == (
        "Save cancelled."
    )

    def unavailable(initial):
        raise file_dialog.DialogUnavailable("no file dialog available")

    monkeypatch.setattr(file_dialog, "ask_save_path", unavailable)
    status = app.save_as(image_path, "dummy", "png", 90, "", False, *defaults(app))[1]
    assert "Save to" in status


# --- presets, compare, batch -----------------------------------------------------------


def test_preset_roundtrip_sets_widgets(app):
    values = defaults(app)
    values[0] = 64  # dummy size
    values[-5] = 1.5  # saturation
    dd, status = app.save_preset("Strong", "dummy", *values)
    assert dd["choices"] and "Saved" in status
    out = app.load_preset("Strong")
    model, *rest, status = out
    assert model == "dummy" and "Loaded" in status
    saturation = len(app.slots) + 1 + len(app.rslots)
    assert rest[0] == 64 and rest[saturation] == 1.5


def test_preset_errors(app):
    assert "Pick" in app.load_preset(None)[-1]
    assert "Can't load" in app.load_preset("missing")[-1]
    assert "Can't save" in app.save_preset("../x", "dummy", *defaults(app))[1]


def test_compare_runs_each_model_once(app, image_path, registry):
    other = type("Other", (DummyModel,), {"id": "other", "display_name": "Other"})
    registry.register(other)
    app2 = App(registry=registry, output_dir=app.output_dir, config_dir=app.config_dir)
    vals = defaults(app2)
    *_, (gallery, status) = app2.compare(image_path, ["dummy", "other"], "cpu", *vals)
    assert [c for _, c in gallery] == [
        "Dummy (size=32, a=10.0, b=-20.0)",
        "Other (size=32, a=10.0, b=-20.0)",
    ]
    assert gallery[0][0].shape == (40, 50, 3) and "2 models" in status
    *_, (_, _) = app2.compare(image_path, ["dummy"], "cpu", *vals)
    assert app2.session.inference_count == 2  # second compare hit the cache
    app2.worker.shutdown()


def test_compare_needs_models(app, image_path):
    with pytest.raises(gr.Error):
        next(app.compare(image_path, [], "cpu", *defaults(app)))


@pytest.mark.filterwarnings("ignore:.*FAILED.*broken.png")  # surfaced as a UI warning
def test_batch_writes_skips_and_reports_failures(app, tmp_path):
    files = []
    for name in ("a", "b"):
        p = tmp_path / f"{name}.png"
        Image.fromarray(np.full((20, 30), 120, np.uint8)).save(p)
        files.append(str(p))
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"nope")
    out = tmp_path / "batch-out"
    out.mkdir()
    (out / "b_colorized.jpg").write_bytes(b"existing")
    *_, (gallery, status) = app.batch(
        [*files, str(broken)], "dummy", "cpu", "jpg", 90, str(out), False, *defaults(app)
    )
    assert Image.open(out / "a_colorized.jpg").format == "JPEG"
    assert (out / "b_colorized.jpg").read_bytes() == b"existing"
    assert "1 colorized, 1 skipped, 1 failed" in status and "broken.png" in status
    assert [c for _, c in gallery] == ["a_colorized.jpg"]
    assert app.last_save_dir == out


def test_batch_needs_files_and_folder(app):
    with pytest.raises(gr.Error):
        next(app.batch([], "dummy", "cpu", "png", 90, "/tmp", False, *defaults(app)))
    with pytest.raises(gr.Error):
        next(app.batch(["x.png"], "dummy", "cpu", "png", 90, " ", False, *defaults(app)))


def test_parse_color_formats():
    from colorizer.ui.gradio_app import parse_color

    assert parse_color("#3c78c8") == (60, 120, 200)
    assert parse_color("rgba(60.2, 120, 199.6, 1)") == (60, 120, 200)
    with pytest.raises(ValueError):
        parse_color("blue")


def test_add_and_remove_points(registry, tmp_path):
    hinted = type(
        "Hinted",
        (DummyModel,),
        {"id": "hinted", "params": (*DummyModel.params, Param("hints", "points", []))},
    )
    registry.register(hinted)
    app = App(registry=registry, output_dir=tmp_path / "o", config_dir=tmp_path / "c")
    assert app.has_points("hinted") and not app.has_points("dummy")
    value = app.add_point("[]", 10, 20, "#ff0000")
    value = app.add_point(value, 30, 40, "#00ff00")
    assert json.loads(value) == [[10, 20, [255, 0, 0]], [30, 40, [0, 255, 0]]]
    assert json.loads(app.remove_point(value, 0)) == [[30, 40, [0, 255, 0]]]
    app.worker.shutdown()


# --- restoration ---------------------------------------------------------------------------


def set_rparam(app, values, rid, name, value):
    """Set a restorer widget value in a flat value list."""
    index = next(i for i, s in enumerate(app.rslots) if (s.model_id, s.param.name) == (rid, name))
    values[len(app.slots) + 1 + index] = value


def test_every_registered_restorer_gets_widgets():
    from colorizer.core.restore import RESTORE_PARAMS

    app = App(registry=Registry())
    try:
        assert set(app.restorers) == {"descratch", "realesrgan", "scunet", "codeformer"}
        for rid, rcls in app.restorers.items():
            names = [s.param.name for s in app.rslots if s.model_id == rid]
            assert names == [p.name for p in (*rcls.params, *RESTORE_PARAMS)]
        assert app.coming_restorers == [
            "SwinIR (not implemented yet)",
            "SeedVR2 (not implemented yet)",
        ]
        assert "non-commercial" in app.restorer_info("codeformer")
        app.build()  # every widget can be constructed
    finally:
        app.worker.shutdown()


def test_restore_views_and_caching(app, image_path):
    values = defaults(app, restore=["dummy_restore"])
    set_rparam(app, values, "dummy_restore", "scale", 2)
    (left, right), _, status = run_colorize(app, image_path, values)
    assert left.shape == right.shape == (80, 100, 3)  # restored 2x; original upscaled to match
    assert "Dummy restorer → Dummy" in status
    assert app.session.restore_count == 1 and app.session.inference_count == 1

    pairs = {}
    for view in VIEWS:
        pair, _, status = app.rerender(image_path, "dummy", "png", 95, view, *values)
        assert status.startswith("Updated")
        pairs[view] = pair
    orig_restored = pairs["Original | Restored"]
    assert all(np.ptp(img.astype(int), axis=2).max() <= 1 for img in orig_restored)  # both gray
    assert np.ptp(pairs["Restored | Colorized"][1].astype(int), axis=2).max() > 5
    assert app.session.restore_count == 1 and app.session.inference_count == 1

    # Colorizer change: restoration reused, model runs again.
    values[0] = 64
    run_colorize(app, image_path, values)
    assert app.session.restore_count == 1 and app.session.inference_count == 2
    # Restore change: both run again; before Colorize, rerender asks for it.
    set_rparam(app, values, "dummy_restore", "offset", 0.1)
    _, _, status = app.rerender(image_path, "dummy", "png", 95, VIEWS[0], *values)
    assert "Colorize" in status
    run_colorize(app, image_path, values)
    assert app.session.restore_count == 2 and app.session.inference_count == 3


def test_restore_errors(app, image_path):
    values = defaults(app, restore=["nope"])
    with pytest.raises(gr.Error, match="unknown restorer"):
        run_colorize(app, image_path, values)
    values = defaults(app, restore=["dummy_restore"])
    set_rparam(app, values, "dummy_restore", "blend", 2.0)
    with pytest.raises(gr.Error):
        run_colorize(app, image_path, values)


def test_compare_restores_once(app, image_path):
    values = defaults(app, restore=["dummy_restore"])
    set_rparam(app, values, "dummy_restore", "scale", 2)
    *_, (gallery, _) = app.compare(image_path, ["dummy"], "cpu", *values)
    assert gallery[0][0].shape == (80, 100, 3) and app.session.restore_count == 1
    run_colorize(app, image_path, values)
    assert app.session.restore_count == 1 and app.session.inference_count == 1


def test_preset_roundtrip_with_restore(app):
    values = defaults(app, restore=["dummy_restore"])
    set_rparam(app, values, "dummy_restore", "scale", 3)
    app.save_preset("Up", "dummy", *values)
    _, *rest, status = app.load_preset("Up")
    assert "Loaded" in status
    n = len(app.slots)
    assert rest[n] == ["dummy_restore"]
    scale_index = next(i for i, s in enumerate(app.rslots) if s.param.name == "scale")
    assert rest[n + 1 + scale_index] == 3
