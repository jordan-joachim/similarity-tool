"""Tests for the 2x4 thumbnail grid: cell contents, aspect-preserving
downscaled previews, checkbox/click/keyboard selection, Select All / Select
None, and per-result selection state that does not leak across result rows.

The grid is exercised both directly (calling ``_load_grid`` with synthetic
photos) and through a real scan so the full scan -> result list -> grid path
is covered. Thumbnails are downscaled on a background thread; tests pump the
GLib main context until the thread finishes.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk

from similarity_tool.config import Config
from similarity_tool.gui import MainWindow, _human_size
from similarity_tool.models import BlurCandidate, Cluster, PhotoFile


def _make_app() -> Gtk.Application:
    """Create a registered Gtk.Application with a unique ID for one test."""
    app = Gtk.Application(application_id=f"io.github.joachim.similaritytool.test.t{uuid.uuid4().hex}")
    app.register()
    return app


def _make_window(cfg: Config | None = None) -> MainWindow:
    app = _make_app()
    win = MainWindow(app, cfg or Config())
    app.window = win  # type: ignore[attr-defined]
    return win


def _make_image(path: Path, width: int = 32, height: int = 32, seed: int = 0) -> None:
    """Write a small valid JPEG with deterministic content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


def _make_photo(tmp_path: Path, name: str, size: int = 1234) -> PhotoFile:
    """Create a real tiny image file and a PhotoFile pointing at it."""
    path = tmp_path / name
    _make_image(path, width=16, height=16, seed=abs(hash(name)) % 1000)
    return PhotoFile(path=path, relative_path=f"2024/05/{name}", size=size, mtime=0.0)


def _make_photo_root(tmp_path: Path) -> Path:
    """Create a photo root with a single numeric 2024/05 month folder."""
    root = tmp_path / "Bilder"
    (root / "2024" / "05").mkdir(parents=True, exist_ok=True)
    return root


def _select_month(win: MainWindow, year: str, month: str) -> None:
    """Select the (year, month) node in the nav tree."""
    for i in range(win.nav_store.iter_n_children(None)):
        year_iter = win.nav_store.iter_nth_child(None, i)
        if win.nav_store.get_value(year_iter, 0) == year:
            for j in range(win.nav_store.iter_n_children(year_iter)):
                month_iter = win.nav_store.iter_nth_child(year_iter, j)
                if win.nav_store.get_value(month_iter, 0) == month:
                    win.nav_selection.select_iter(month_iter)
                    return
    raise AssertionError(f"month {year}/{month} not in the nav tree")


def _pump_until_idle(win: MainWindow, timeout: float = 60.0) -> None:
    """Run the default GLib main context until the scan thread finishes."""
    import gi

    gi.require_version("GLib", "2.0")
    from gi.repository import GLib

    deadline = time.monotonic() + timeout
    ctx = GLib.MainContext.default()
    while time.monotonic() < deadline:
        while ctx.pending():
            ctx.iteration(False)
        thread = win._scan_thread
        if thread is None or not thread.is_alive():
            break
        time.sleep(0.01)
    while ctx.pending():
        ctx.iteration(False)


def _pump_until_thumbnails(win: MainWindow, timeout: float = 10.0) -> None:
    """Wait for the thumbnail worker thread to finish and idle callbacks to run."""
    import gi

    gi.require_version("GLib", "2.0")
    from gi.repository import GLib

    deadline = time.monotonic() + timeout
    ctx = GLib.MainContext.default()
    while time.monotonic() < deadline:
        while ctx.pending():
            ctx.iteration(False)
        thread = win._thumb_thread
        if thread is None or not thread.is_alive():
            break
        time.sleep(0.01)
    while ctx.pending():
        ctx.iteration(False)


class TestGridStructure:
    def test_each_cell_has_preview_filename_size_checkbox(self):
        win = _make_window()
        assert len(win._grid_cells) == 8
        for cell in win._grid_cells:
            assert isinstance(cell.picture, Gtk.Picture)
            assert isinstance(cell.name_label, Gtk.Label)
            assert isinstance(cell.size_label, Gtk.Label)
            assert isinstance(cell.checkbox, Gtk.CheckButton)

    def test_grid_is_two_rows_four_columns(self):
        win = _make_window()
        grid = win.thumb_grid
        layout = grid.get_layout_manager()
        positions = set()
        for cell in win._grid_cells:
            child = layout.get_layout_child(cell)
            positions.add((child.get_property("row"), child.get_property("column")))
        assert positions == {(r, c) for r in range(2) for c in range(4)}


class TestGridPopulation:
    def test_populated_cell_shows_filename_and_size(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg", size=1000 + i) for i in range(3)]
        win._load_grid(1, photos)
        cell = win._grid_cells[0]
        assert cell.photo is not None
        assert cell.name_label.get_text() == "img0.jpg"
        assert cell.size_label.get_text() == "1000 B"
        assert cell.checkbox.get_sensitive() is True

    def test_empty_cells_are_placeholders(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(2)]
        win._load_grid(1, photos)
        for index in range(2, 8):
            cell = win._grid_cells[index]
            assert cell.photo is None
            assert cell.name_label.get_text() == ""
            assert cell.size_label.get_text() == ""
            assert cell.checkbox.get_sensitive() is False

    def test_grid_never_shows_more_than_eight_cells(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(10)]
        win._load_grid(1, photos)
        populated = [cell for cell in win._grid_cells if cell.photo is not None]
        assert len(populated) == 8

    def test_thumbnail_preserves_aspect_ratio(self, tmp_path):
        path = tmp_path / "wide.jpg"
        _make_image(path, width=800, height=400)
        photo = PhotoFile(path=path, relative_path="2024/05/wide.jpg", size=1000, mtime=0.0)
        win = _make_window()
        win._load_grid(1, [photo])
        _pump_until_thumbnails(win)
        texture = win._grid_cells[0].picture.get_paintable()
        assert texture is not None
        ratio = texture.get_width() / texture.get_height()
        assert ratio == pytest.approx(2.0, abs=0.05)

    def test_thumbnail_downscaled_to_fit_cell(self, tmp_path):
        path = tmp_path / "big.jpg"
        _make_image(path, width=2000, height=1000)
        photo = PhotoFile(path=path, relative_path="2024/05/big.jpg", size=1000, mtime=0.0)
        win = _make_window()
        win._load_grid(1, [photo])
        _pump_until_thumbnails(win)
        texture = win._grid_cells[0].picture.get_paintable()
        assert texture is not None
        assert max(texture.get_width(), texture.get_height()) <= 200

    def test_corrupt_image_does_not_crash_grid(self, tmp_path):
        path = tmp_path / "broken.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe0 truncated")
        photo = PhotoFile(path=path, relative_path="2024/05/broken.jpg", size=1000, mtime=0.0)
        win = _make_window()
        win._load_grid(1, [photo])
        _pump_until_thumbnails(win)
        cell = win._grid_cells[0]
        assert cell.photo is not None
        assert cell.name_label.get_text() == "broken.jpg"
        # The preview may be empty; the grid must not crash.


class TestSelection:
    def test_checkbox_toggles_selection(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(4)]
        win._load_grid(1, photos)
        cell = win._grid_cells[0]
        cell.checkbox.set_active(True)
        assert 0 in win._selection
        cell.checkbox.set_active(False)
        assert 0 not in win._selection

    def test_multiple_cells_can_be_selected(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(4)]
        win._load_grid(1, photos)
        for index in (0, 1, 2):
            win._grid_cells[index].checkbox.set_active(True)
        assert win._selection == {0, 1, 2}
        assert all(win._grid_cells[i].checkbox.get_active() for i in (0, 1, 2))

    def test_select_all_selects_populated_cells(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(3)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        assert win._selection == {0, 1, 2}
        assert all(win._grid_cells[i].checkbox.get_active() for i in range(3))
        # Empty cells stay unselected.
        assert not win._grid_cells[3].checkbox.get_active()

    def test_select_none_clears_selection(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(4)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        win._on_select_none_clicked(None)
        assert win._selection == set()
        assert not any(cell.checkbox.get_active() for cell in win._grid_cells)

    def test_click_on_cell_toggles_selection(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(4)]
        win._load_grid(1, photos)
        cell = win._grid_cells[0]
        cell.gesture.emit("pressed", 1, 10.0, 10.0)
        assert 0 in win._selection
        cell.gesture.emit("pressed", 1, 10.0, 10.0)
        assert 0 not in win._selection

    def test_keyboard_space_toggles_focused_cell(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(4)]
        win._load_grid(1, photos)
        win._focused_cell_index = 0
        win._on_grid_key_pressed(None, Gdk.KEY_space, 0, 0)
        assert 0 in win._selection
        win._on_grid_key_pressed(None, Gdk.KEY_space, 0, 0)
        assert 0 not in win._selection

    def test_keyboard_a_selects_all(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(4)]
        win._load_grid(1, photos)
        win._on_grid_key_pressed(None, Gdk.KEY_a, 0, 0)
        assert win._selection == {0, 1, 2, 3}

    def test_keyboard_n_selects_none(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(4)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        win._on_grid_key_pressed(None, Gdk.KEY_n, 0, 0)
        assert win._selection == set()


class TestPerResultSelection:
    def test_selection_does_not_leak_between_results(self, tmp_path):
        win = _make_window()
        photos_a = [_make_photo(tmp_path, f"a{i}.jpg") for i in range(4)]
        photos_b = [_make_photo(tmp_path, f"b{i}.jpg") for i in range(4)]
        win._load_grid(1, photos_a)
        win._grid_cells[0].checkbox.set_active(True)
        win._grid_cells[1].checkbox.set_active(True)
        assert win._selection == {0, 1}
        # Switch to B: no prior selections.
        win._load_grid(2, photos_b)
        assert win._selection == set()
        assert not any(cell.checkbox.get_active() for cell in win._grid_cells)
        # Switch back to A: prior selections are restored.
        win._load_grid(1, photos_a)
        assert win._selection == {0, 1}
        assert win._grid_cells[0].checkbox.get_active() is True
        assert win._grid_cells[1].checkbox.get_active() is True

    def test_switching_result_rows_does_not_leak_selection(self, tmp_path):
        root = _make_photo_root(tmp_path)
        for i in range(4):
            _make_image(root / "2024" / "05" / f"burst{i}.jpg", seed=10)
        for i in range(2):
            _make_image(root / "2024" / "05" / f"pair{i}.jpg", seed=20)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        # Row 0 is auto-selected (cluster 1, 4 images).
        win._grid_cells[0].checkbox.set_active(True)
        win._grid_cells[1].checkbox.set_active(True)
        assert win._selection == {0, 1}
        # Select row 1 (cluster 2, 2 images): grid shows no prior selections.
        win.result_list.select_row(win.result_list.get_row_at_index(1))
        assert win._selection == set()
        assert not any(cell.checkbox.get_active() for cell in win._grid_cells)
        assert win._grid_cells[0].name_label.get_text() == "pair0.jpg"
        # Back to row 0: prior selections are restored.
        win.result_list.select_row(win.result_list.get_row_at_index(0))
        assert win._selection == {0, 1}
        assert win._grid_cells[0].checkbox.get_active() is True
        assert win._grid_cells[1].checkbox.get_active() is True


class TestBlurGrid:
    def test_blur_candidate_shows_score_tooltip(self, tmp_path):
        win = _make_window()
        photo = _make_photo(tmp_path, "blurry.jpg")
        candidate = BlurCandidate(photo=photo, score=42.5, percentile=7.0)
        win._load_grid(1, [photo], tooltips=["Blur score 42.5, 7%"])
        cell = win._grid_cells[0]
        assert cell.photo is not None
        assert cell.name_label.get_text() == "blurry.jpg"
        assert "42.5" in (cell.get_tooltip_text() or "")


class TestSelectButtonSensitivity:
    def test_select_buttons_enabled_when_grid_populated(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(2)]
        win._load_grid(1, photos)
        assert win.select_all_button.get_sensitive() is True
        assert win.select_none_button.get_sensitive() is False
        win._on_select_all_clicked(None)
        assert win.select_none_button.get_sensitive() is True

    def test_select_buttons_disabled_when_grid_empty(self, tmp_path):
        win = _make_window()
        win._clear_grid()
        assert win.select_all_button.get_sensitive() is False
        assert win.select_none_button.get_sensitive() is False


class TestHumanSize:
    def test_human_size_formats_bytes(self):
        assert _human_size(0) == "0 B"
        assert _human_size(999) == "999 B"
        assert _human_size(1024) == "1.0 KB"
        assert _human_size(1536) == "1.5 KB"
        assert _human_size(5 * 1024 * 1024) == "5.0 MB"
        assert _human_size(2 * 1024 * 1024 * 1024) == "2.0 GB"


class TestKeyboardFocus:
    def test_arrow_keys_move_focus(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(8)]
        win._load_grid(1, photos)
        win._focused_cell_index = 0
        win._on_grid_key_pressed(None, Gdk.KEY_Right, 0, 0)
        assert win._focused_cell_index == 1
        win._on_grid_key_pressed(None, Gdk.KEY_Down, 0, 0)
        assert win._focused_cell_index == 5
        win._on_grid_key_pressed(None, Gdk.KEY_Left, 0, 0)
        assert win._focused_cell_index == 4
        win._on_grid_key_pressed(None, Gdk.KEY_Up, 0, 0)
        assert win._focused_cell_index == 0

    def test_arrow_keys_clamp_at_grid_edges(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(8)]
        win._load_grid(1, photos)
        win._focused_cell_index = 0
        win._on_grid_key_pressed(None, Gdk.KEY_Left, 0, 0)
        assert win._focused_cell_index == 0
        win._on_grid_key_pressed(None, Gdk.KEY_Up, 0, 0)
        assert win._focused_cell_index == 0
        win._focused_cell_index = 7
        win._on_grid_key_pressed(None, Gdk.KEY_Right, 0, 0)
        assert win._focused_cell_index == 7
        win._on_grid_key_pressed(None, Gdk.KEY_Down, 0, 0)
        assert win._focused_cell_index == 7


class TestScanToGrid:
    def test_scan_populates_grid_with_thumbnails(self, tmp_path):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        _pump_until_thumbnails(win)
        cell = win._grid_cells[0]
        assert cell.photo is not None
        assert cell.name_label.get_text() == "a.jpg"
        assert cell.picture.get_paintable() is not None
