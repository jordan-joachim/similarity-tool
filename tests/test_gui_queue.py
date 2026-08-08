"""Tests for the shared deletion queue: Add to Queue, duplicate rejection,
selection clearing, the queue review tab (thumbnails, filenames, sizes,
source-mode indicators, aggregate count/size), per-item removal, Discard
Queue confirmation, Execute Queue confirmation, batch move to trash, and
queue survival across mode/month navigation.

The queue is exercised both directly (calling the queue methods with
synthetic photos) and through real scans so the full scan -> grid -> queue
path is covered. Confirmation dialogs are monkeypatched so tests run headless.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from similarity_tool.config import Config
from similarity_tool.gui import MainWindow
from similarity_tool.models import BlurCandidate, Cluster, PhotoFile, QueueItem


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


def _pump_until_execution(win: MainWindow, timeout: float = 30.0) -> None:
    """Wait for the trash execution thread to finish and idle callbacks to run."""
    import gi

    gi.require_version("GLib", "2.0")
    from gi.repository import GLib

    deadline = time.monotonic() + timeout
    ctx = GLib.MainContext.default()
    while time.monotonic() < deadline:
        while ctx.pending():
            ctx.iteration(False)
        thread = win._execution_thread
        if thread is None or not thread.is_alive():
            break
        time.sleep(0.01)
    while ctx.pending():
        ctx.iteration(False)


def _queue_rows(win: MainWindow) -> list[Gtk.FlowBoxChild]:
    """Return the FlowBox children currently in the queue tab."""
    rows: list[Gtk.FlowBoxChild] = []
    child = win.queue_flow.get_first_child()
    while child is not None:
        rows.append(child)
        child = child.get_next_sibling()
    return rows


def _queue_photos(win: MainWindow) -> list[PhotoFile]:
    """Return the photos currently staged in the queue, in display order."""
    return [row.item.photo for row in _queue_rows(win)]


def _queue_modes(win: MainWindow) -> list[str]:
    """Return the source modes of the currently staged queue items."""
    return [row.item.mode for row in _queue_rows(win)]


def _log_text(win: MainWindow) -> str:
    buf = win.log_buffer
    return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)


def _today() -> str:
    """Return today's date in the YYYY-MM-DD format used by the trash folder."""
    return datetime.now().strftime("%Y-%m-%d")


class TestAddToQueue:
    def test_add_to_queue_stages_selected_photos(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg", size=1000 + i) for i in range(3)]
        win._load_grid(1, photos)
        win._grid_cells[0].checkbox.set_active(True)
        win._grid_cells[2].checkbox.set_active(True)
        win._on_add_to_queue_clicked(None)
        assert _queue_photos(win) == [photos[0], photos[2]]
        assert win.execute_queue_button.get_sensitive() is True

    def test_add_to_queue_clears_grid_selection(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(4)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        assert win._selection == {0, 1, 2, 3}
        win._on_add_to_queue_clicked(None)
        assert win._selection == set()
        assert not any(cell.checkbox.get_active() for cell in win._grid_cells)
        assert win.status_label.get_text() == "0 selected"

    def test_add_to_queue_rejects_duplicates(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(2)]
        win._load_grid(1, photos)
        win._grid_cells[0].checkbox.set_active(True)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 1
        # Re-offer the same file from a different result row.
        win._load_grid(2, photos)
        win._grid_cells[0].checkbox.set_active(True)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 1
        assert _queue_photos(win) == [photos[0]]

    def test_add_to_queue_is_noop_when_nothing_selected(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(2)]
        win._load_grid(1, photos)
        win._on_add_to_queue_clicked(None)
        assert _queue_rows(win) == []
        assert win.execute_queue_button.get_sensitive() is False

    def test_add_to_queue_button_insensitive_without_selection(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(2)]
        win._load_grid(1, photos)
        assert win.add_to_queue_button.get_sensitive() is False
        win._grid_cells[0].checkbox.set_active(True)
        assert win.add_to_queue_button.get_sensitive() is True
        win._grid_cells[0].checkbox.set_active(False)
        assert win.add_to_queue_button.get_sensitive() is False

    def test_add_to_queue_from_similarity_scan(self, tmp_path):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        win._grid_cells[0].checkbox.set_active(True)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 1
        assert _queue_modes(win) == ["similarity"]

    def test_add_to_queue_from_blur_scan(self, tmp_path):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=2)
        _make_image(root / "2024" / "05" / "c.jpg", seed=3)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.mode_selector.set_selected(1)
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        win._grid_cells[0].checkbox.set_active(True)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 1
        assert _queue_modes(win) == ["blur"]


class TestQueueTab:
    def test_queue_tab_shows_thumbnail_filename_size_and_mode(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, "img0.jpg", size=2048)]
        win._load_grid(1, photos)
        win._grid_cells[0].checkbox.set_active(True)
        win._on_add_to_queue_clicked(None)
        _pump_until_thumbnails(win)
        row = _queue_rows(win)[0]
        assert row.item.photo == photos[0]
        assert row.name_label.get_text() == "img0.jpg"
        assert row.size_label.get_text() == "2.0 KB"
        assert row.mode_label.get_text() == "Similarity"
        assert row.picture.get_paintable() is not None

    def test_queue_shows_aggregate_count_and_size(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg", size=1024) for i in range(3)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        assert win.queue_count_label.get_text() == "Queue: 3 items (3.0 KB)"

    def test_queue_count_resets_when_empty(self, tmp_path):
        win = _make_window()
        assert win.queue_count_label.get_text() == "Queue: 0 items (0 B)"
        assert win.execute_queue_button.get_sensitive() is False
        assert win.discard_queue_button.get_sensitive() is False

    def test_queue_tab_has_flowbox_with_thumbnails(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(5)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 5


class TestRemoveFromQueue:
    def test_remove_single_item(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(3)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 3
        _queue_rows(win)[1].remove_button.emit("clicked")
        assert _queue_photos(win) == [photos[0], photos[2]]
        assert win.queue_count_label.get_text() == "Queue: 2 items (2.4 KB)"
        # The original file is untouched.
        assert photos[1].path.exists()

    def test_remove_last_item_disables_execute(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, "only.jpg")]
        win._load_grid(1, photos)
        win._grid_cells[0].checkbox.set_active(True)
        win._on_add_to_queue_clicked(None)
        assert win.execute_queue_button.get_sensitive() is True
        _queue_rows(win)[0].remove_button.emit("clicked")
        assert _queue_rows(win) == []
        assert win.execute_queue_button.get_sensitive() is False
        assert win.discard_queue_button.get_sensitive() is False


class TestDiscardQueue:
    def test_discard_clears_queue_after_confirmation(self, tmp_path, monkeypatch):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(2)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 2
        monkeypatch.setattr(win, "_confirm_discard_queue", lambda: True)
        win._on_discard_queue_clicked(None)
        assert _queue_rows(win) == []
        assert win.queue_count_label.get_text() == "Queue: 0 items (0 B)"
        assert win.execute_queue_button.get_sensitive() is False
        # No files were moved.
        assert all(photo.path.exists() for photo in photos)

    def test_discard_cancel_keeps_queue(self, tmp_path, monkeypatch):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(2)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        monkeypatch.setattr(win, "_confirm_discard_queue", lambda: False)
        win._on_discard_queue_clicked(None)
        assert len(_queue_rows(win)) == 2
        assert win.execute_queue_button.get_sensitive() is True

    def test_discard_with_empty_queue_is_noop(self, tmp_path):
        win = _make_window()
        win._on_discard_queue_clicked(None)
        assert _queue_rows(win) == []


class TestExecuteQueue:
    def test_execute_moves_files_to_trash_and_clears_queue(self, tmp_path, monkeypatch):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        trash_root = tmp_path / "trash"
        cfg = Config(
            photo_root=str(root),
            cache_path=str(tmp_path / "cache" / "h.sqlite3"),
            trash_root=str(trash_root),
        )
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 2
        monkeypatch.setattr(win, "_confirm_execute_queue", lambda: True)
        win._on_execute_queue_clicked(None)
        _pump_until_execution(win)
        assert _queue_rows(win) == []
        assert win.queue_count_label.get_text() == "Queue: 0 items (0 B)"
        assert win.execute_queue_button.get_sensitive() is False
        # Files moved to a dated trash folder, not deleted.
        assert not (root / "2024" / "05" / "a.jpg").exists()
        assert not (root / "2024" / "05" / "b.jpg").exists()
        dated = trash_root / _today()
        assert dated.is_dir()
        moved = list(dated.rglob("*.jpg"))
        assert len(moved) == 2
        # JSON log written next to the dated folder.
        log_path = dated / "trash.log.json"
        assert log_path.exists()
        import json

        data = json.loads(log_path.read_text())
        assert len(data["entries"]) == 2
        assert all(e["mode"] == "similarity" for e in data["entries"])

    def test_execute_cancel_leaves_queue_and_files(self, tmp_path, monkeypatch):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        trash_root = tmp_path / "trash"
        cfg = Config(
            photo_root=str(root),
            cache_path=str(tmp_path / "cache" / "h.sqlite3"),
            trash_root=str(trash_root),
        )
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        monkeypatch.setattr(win, "_confirm_execute_queue", lambda: False)
        win._on_execute_queue_clicked(None)
        assert len(_queue_rows(win)) == 2
        assert (root / "2024" / "05" / "a.jpg").exists()
        assert (root / "2024" / "05" / "b.jpg").exists()
        assert not (trash_root / _today()).exists()

    def test_execute_with_empty_queue_is_noop(self, tmp_path, monkeypatch):
        win = _make_window()
        monkeypatch.setattr(win, "_confirm_execute_queue", lambda: True)
        win._on_execute_queue_clicked(None)
        assert _queue_rows(win) == []
        assert "trash" not in _log_text(win).lower()

    def test_execute_mixed_modes_writes_per_item_mode(self, tmp_path, monkeypatch):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        _make_image(root / "2024" / "05" / "c.jpg", seed=2)
        _make_image(root / "2024" / "05" / "d.jpg", seed=3)
        trash_root = tmp_path / "trash"
        cfg = Config(
            photo_root=str(root),
            cache_path=str(tmp_path / "cache" / "h.sqlite3"),
            trash_root=str(trash_root),
        )
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        # Stage one similarity item.
        win._grid_cells[0].checkbox.set_active(True)
        win._on_add_to_queue_clicked(None)
        # Switch to Blur and stage one candidate.
        win.mode_selector.set_selected(1)
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        win._grid_cells[0].checkbox.set_active(True)
        win._on_add_to_queue_clicked(None)
        assert _queue_modes(win) == ["similarity", "blur"]
        monkeypatch.setattr(win, "_confirm_execute_queue", lambda: True)
        win._on_execute_queue_clicked(None)
        _pump_until_execution(win)
        assert _queue_rows(win) == []
        import json

        data = json.loads((trash_root / _today() / "trash.log.json").read_text())
        modes = {e["mode"] for e in data["entries"]}
        assert modes == {"similarity", "blur"}

    def test_execute_logs_and_removes_moved_files_from_view(self, tmp_path, monkeypatch):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        trash_root = tmp_path / "trash"
        cfg = Config(
            photo_root=str(root),
            cache_path=str(tmp_path / "cache" / "h.sqlite3"),
            trash_root=str(trash_root),
        )
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        monkeypatch.setattr(win, "_confirm_execute_queue", lambda: True)
        win._on_execute_queue_clicked(None)
        _pump_until_execution(win)
        text = _log_text(win)
        assert "Moved 2 file(s) to trash" in text
        assert "trash.log.json" in text

    def test_executed_files_removed_from_review_grid(self, tmp_path, monkeypatch):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        trash_root = tmp_path / "trash"
        cfg = Config(
            photo_root=str(root),
            cache_path=str(tmp_path / "cache" / "h.sqlite3"),
            trash_root=str(trash_root),
        )
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        # Stage only the first grid cell.
        moved_name = win._grid_cells[0].name_label.get_text()
        win._grid_cells[0].checkbox.set_active(True)
        win._on_add_to_queue_clicked(None)
        monkeypatch.setattr(win, "_confirm_execute_queue", lambda: True)
        win._on_execute_queue_clicked(None)
        _pump_until_execution(win)
        # The moved file no longer appears in the grid; the remaining member
        # stays visible (VAL-QUEUE-020 / VAL-SIM-020).
        labels = [cell.name_label.get_text() for cell in win._grid_cells if cell.photo is not None]
        assert moved_name not in labels
        assert len(labels) == 1

    def test_empty_cluster_removed_from_result_list(self, tmp_path, monkeypatch):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        trash_root = tmp_path / "trash"
        cfg = Config(
            photo_root=str(root),
            cache_path=str(tmp_path / "cache" / "h.sqlite3"),
            trash_root=str(trash_root),
        )
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        assert win.result_list.get_row_at_index(0) is not None
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        monkeypatch.setattr(win, "_confirm_execute_queue", lambda: True)
        win._on_execute_queue_clicked(None)
        _pump_until_execution(win)
        # The cluster became empty and was removed from the result list
        # (VAL-XFL-016).
        assert win.result_list.get_row_at_index(0) is None
        assert not any(cell.photo is not None for cell in win._grid_cells)


class TestQueueSurvival:
    def test_queue_survives_mode_switch(self, tmp_path):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(2)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 2
        win.mode_selector.set_selected(1)
        assert len(_queue_rows(win)) == 2
        assert win.execute_queue_button.get_sensitive() is True
        win.mode_selector.set_selected(0)
        assert len(_queue_rows(win)) == 2

    def test_queue_survives_month_change(self, tmp_path):
        root = _make_photo_root(tmp_path)
        (root / "2024" / "06").mkdir(parents=True)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 2
        _select_month(win, "2024", "06")
        assert len(_queue_rows(win)) == 2
        assert win.execute_queue_button.get_sensitive() is True

    def test_queue_survives_result_row_navigation(self, tmp_path):
        win = _make_window()
        photos_a = [_make_photo(tmp_path, f"a{i}.jpg") for i in range(2)]
        photos_b = [_make_photo(tmp_path, f"b{i}.jpg") for i in range(2)]
        win._load_grid(1, photos_a)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        assert len(_queue_rows(win)) == 2
        win._load_grid(2, photos_b)
        assert len(_queue_rows(win)) == 2
        assert win.execute_queue_button.get_sensitive() is True


class TestConfirmationDialogs:
    def test_confirm_execute_dialog_uses_alert_dialog(self, tmp_path, monkeypatch):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg", size=1024) for i in range(2)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        captured: dict = {}

        def fake_choose(self, parent, cancellable, callback, user_data):
            captured["message"] = self.get_message()
            captured["detail"] = self.get_detail()
            captured["buttons"] = self.get_buttons()
            captured["cancel"] = self.get_cancel_button()
            captured["default"] = self.get_default_button()
            return 1  # Confirm

        monkeypatch.setattr(Gtk.AlertDialog, "choose", fake_choose)
        win._on_execute_queue_clicked(None)
        assert "2" in captured["message"]
        assert "2.0 KB" in captured["detail"]
        assert captured["buttons"] == ["Cancel", "Move to Trash"]
        assert captured["cancel"] == 0
        assert captured["default"] == 1

    def test_confirm_discard_dialog_uses_alert_dialog(self, tmp_path, monkeypatch):
        win = _make_window()
        photos = [_make_photo(tmp_path, f"img{i}.jpg") for i in range(2)]
        win._load_grid(1, photos)
        win._on_select_all_clicked(None)
        win._on_add_to_queue_clicked(None)
        captured: dict = {}

        def fake_choose(self, parent, cancellable, callback, user_data):
            captured["message"] = self.get_message()
            captured["buttons"] = self.get_buttons()
            captured["cancel"] = self.get_cancel_button()
            return 0  # Cancel

        monkeypatch.setattr(Gtk.AlertDialog, "choose", fake_choose)
        win._on_discard_queue_clicked(None)
        assert "Discard" in captured["message"]
        assert captured["buttons"] == ["Cancel", "Discard Queue"]
        assert captured["cancel"] == 0
        # Cancel leaves the queue intact.
        assert len(_queue_rows(win)) == 2
