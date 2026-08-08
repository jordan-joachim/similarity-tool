"""Tests for the Scan button wiring: starting similarity/blur scans, progress
indicator visibility, result-list population, empty states, invalid month
paths, and mode switching during an active scan.

Scans run on a background thread and marshal results back with
``GLib.idle_add``; the tests pump the default GLib main context until the scan
thread finishes so the idle callbacks run. These tests construct the real
``MainWindow`` widget tree in memory against a temporary photo root, so they
never touch the real archive.
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
from gi.repository import Gtk

from similarity_tool.config import Config
from similarity_tool.gui import MainWindow


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


def _make_image(path: Path, seed: int = 0, size: int = 32) -> None:
    """Write a small valid JPEG with deterministic content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


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
    """Run the default GLib main context until the scan thread finishes.

    The scan thread marshals its result back with ``GLib.idle_add``; pumping
    the main context lets those callbacks run. Returns when no scan thread is
    active and all pending sources have been drained.
    """
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
    # Drain any remaining idle callbacks (e.g. a stale scan's finish).
    while ctx.pending():
        ctx.iteration(False)


def _result_rows(win: MainWindow) -> list[Gtk.ListBoxRow]:
    """Return the rows currently in the result list."""
    rows: list[Gtk.ListBoxRow] = []
    child = win.result_list.get_first_child()
    while child is not None:
        rows.append(child)
        child = child.get_next_sibling()
    return rows


def _grid_labels(win: MainWindow) -> list[str]:
    """Return the text of the labels currently shown in the 2x4 grid cells."""
    labels: list[str] = []
    for cell in win._grid_cells:
        child = cell.get_first_child()
        if isinstance(child, Gtk.Label):
            labels.append(child.get_text())
    return labels


def _log_text(win: MainWindow) -> str:
    buf = win.log_buffer
    return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)


class TestScanStarts:
    def test_scan_button_starts_similarity_scan(self, tmp_path):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)  # identical -> cluster
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        assert win._scanning is True
        assert win._scan_thread is not None
        _pump_until_idle(win)
        assert win._scanning is False
        row = win.result_list.get_row_at_index(0)
        assert row is not None
        assert "Cluster 1" in row.title

    def test_scan_button_starts_blur_scan(self, tmp_path):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=2)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.mode_selector.set_selected(1)  # Blur
        win.scan_button.emit("clicked")
        assert win._scanning is True
        _pump_until_idle(win)
        assert win._scanning is False
        row = win.result_list.get_row_at_index(0)
        assert row is not None
        assert "score" in row.title.lower()

    def test_scan_button_disabled_during_scan_and_reenabled_after(self, tmp_path):
        root = _make_photo_root(tmp_path)
        for i in range(30):
            _make_image(root / "2024" / "05" / f"img{i:02d}.jpg", seed=i)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        assert win.scan_button.get_sensitive() is False
        _pump_until_idle(win)
        assert win.scan_button.get_sensitive() is True

    def test_log_records_scan_start_and_finish(self, tmp_path):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        text = _log_text(win)
        assert "Scanning 2024/05" in text
        assert "Scan finished" in text


class TestProgressIndicator:
    def test_progress_indicator_visible_during_scan(self, tmp_path):
        root = _make_photo_root(tmp_path)
        for i in range(40):
            _make_image(root / "2024" / "05" / f"img{i:02d}.jpg", seed=i)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        # Immediately after clicking, the progress indicator must be active.
        assert win.scan_spinner.get_spinning() is True
        assert win.scan_spinner.get_visible() is True
        assert win.scan_status_label.get_text() != ""
        _pump_until_idle(win)
        assert win.scan_spinner.get_spinning() is False
        assert win.scan_spinner.get_visible() is False

    def test_ui_remains_responsive_during_scan(self, tmp_path):
        """The scan runs on a background thread; the main thread keeps
        processing GTK events while it is in progress."""
        import gi

        gi.require_version("GLib", "2.0")
        from gi.repository import GLib

        root = _make_photo_root(tmp_path)
        for i in range(40):
            _make_image(root / "2024" / "05" / f"img{i:02d}.jpg", seed=i)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        ctx = GLib.MainContext.default()
        pumped = 0
        while win._scanning and pumped < 2000:
            while ctx.pending():
                ctx.iteration(False)
                pumped += 1
            time.sleep(0.005)
        assert pumped > 0, "main loop could not process events during the scan"
        _pump_until_idle(win)


class TestResultPopulation:
    def test_similarity_result_list_has_one_row_per_cluster(self, tmp_path):
        root = _make_photo_root(tmp_path)
        for i in range(4):
            _make_image(root / "2024" / "05" / f"burst{i}.jpg", seed=10)
        for i in range(2):
            _make_image(root / "2024" / "05" / f"pair{i}.jpg", seed=20)
        _make_image(root / "2024" / "05" / "unique.jpg", seed=99)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        rows = _result_rows(win)
        assert len(rows) == 2
        assert "Cluster 1 (4 images)" in rows[0].title
        assert "Cluster 2 (2 images)" in rows[1].title

    def test_blur_result_list_has_candidate_rows_with_scores(self, tmp_path):
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
        rows = _result_rows(win)
        assert len(rows) == 1  # bottom 10% of 3 images = 1 candidate
        assert "score" in rows[0].title.lower()

    def test_selecting_result_row_populates_grid(self, tmp_path):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        # The first row is auto-selected; the grid shows the cluster members.
        labels = _grid_labels(win)
        assert "a.jpg" in labels
        assert "b.jpg" in labels

    def test_corrupt_image_does_not_crash_scan(self, tmp_path):
        root = _make_photo_root(tmp_path)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        (root / "2024" / "05" / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 truncated")
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        assert win._scanning is False
        row = win.result_list.get_row_at_index(0)
        assert row is not None
        assert "broken" not in row.title


class TestEmptyStates:
    def test_empty_month_shows_informative_message(self, tmp_path):
        root = _make_photo_root(tmp_path)  # 2024/05 exists but has no images
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        assert win.result_list.get_row_at_index(0) is None
        assert "No images found" in win.result_header.get_text()

    def test_similarity_no_clusters_shows_empty_state(self, tmp_path):
        root = _make_photo_root(tmp_path)
        for i in range(3):
            _make_image(root / "2024" / "05" / f"u{i}.jpg", seed=i)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        assert win.result_list.get_row_at_index(0) is None
        assert "No similar images found" in win.result_header.get_text()

    def test_blur_empty_month_shows_informative_message(self, tmp_path):
        root = _make_photo_root(tmp_path)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.mode_selector.set_selected(1)
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        assert win.result_list.get_row_at_index(0) is None
        assert "No images found" in win.result_header.get_text()


class TestInvalidPaths:
    def test_missing_month_folder_is_surfaced_without_scan(self, tmp_path):
        root = _make_photo_root(tmp_path)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        # Remove the month folder after the tree was built.
        (root / "2024" / "05").rmdir()
        win.scan_button.emit("clicked")
        assert win._scanning is False
        assert win._scan_thread is None
        assert "does not exist" in win.result_header.get_text()
        assert "does not exist" in _log_text(win)

    def test_missing_photo_root_leaves_scan_disabled(self, tmp_path):
        cfg = Config(photo_root=str(tmp_path / "nope"), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        assert win.scan_button.get_sensitive() is False
        # No month can be selected, so clicking Scan does nothing.
        win.scan_button.emit("clicked")
        assert win._scanning is False
        assert win._scan_thread is None


class TestModeSwitchDuringScan:
    def test_mode_switch_during_scan_does_not_crash(self, tmp_path):
        root = _make_photo_root(tmp_path)
        for i in range(40):
            _make_image(root / "2024" / "05" / f"img{i:02d}.jpg", seed=i)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        assert win._scanning is True
        # Switch mode while the scan is running.
        win.mode_selector.set_selected(1)
        assert win.mode == "blur"
        assert win._scanning is False
        assert win.scan_spinner.get_spinning() is False
        # The stale scan's result must be discarded when it finishes.
        _pump_until_idle(win)
        assert win.result_list.get_row_at_index(0) is None
        # A subsequent scan in the new mode completes normally.
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        assert win._scanning is False
        row = win.result_list.get_row_at_index(0)
        assert row is not None
        assert "score" in row.title.lower()

    def test_mode_switch_back_and_forth_during_scan(self, tmp_path):
        root = _make_photo_root(tmp_path)
        # Identical images cluster reliably in Similarity mode.
        for i in range(4):
            _make_image(root / "2024" / "05" / f"img{i:02d}.jpg", seed=7)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        win.mode_selector.set_selected(1)
        win.mode_selector.set_selected(0)
        assert win.mode == "similarity"
        _pump_until_idle(win)
        assert win.result_list.get_row_at_index(0) is None
        # A new similarity scan works after the switches.
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        assert win._scanning is False
        assert win.result_list.get_row_at_index(0) is not None


class TestMonthChangeClearsResults:
    def test_selecting_different_month_clears_results(self, tmp_path):
        root = _make_photo_root(tmp_path)
        (root / "2024" / "06").mkdir(parents=True)
        _make_image(root / "2024" / "05" / "a.jpg", seed=1)
        _make_image(root / "2024" / "05" / "b.jpg", seed=1)
        cfg = Config(photo_root=str(root), cache_path=str(tmp_path / "cache" / "h.sqlite3"))
        win = _make_window(cfg)
        _select_month(win, "2024", "05")
        win.scan_button.emit("clicked")
        _pump_until_idle(win)
        assert win.result_list.get_row_at_index(0) is not None
        # Selecting a different month resets the results (VAL-BLUR-004).
        _select_month(win, "2024", "06")
        assert win.result_list.get_row_at_index(0) is None
        assert "Select a month and press Scan" in win.result_header.get_text()
