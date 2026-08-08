"""Tests for the GTK4 main window layout: regions, mode selector, year/month tree.

These tests construct the real ``MainWindow`` widget tree in memory. Tests that
need a display (allocation checks) are skipped when no GTK display is available.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from similarity_tool.config import Config
from similarity_tool.gui import MainWindow

HAS_DISPLAY = Gtk.init_check()


def _make_app() -> Gtk.Application:
    """Create a registered Gtk.Application with a unique ID for one test.

    The final ID element must start with a letter (GApplication ID rule), so a
    fixed ``t`` prefix is used before the hex suffix.
    """
    app = Gtk.Application(application_id=f"io.github.joachim.similaritytool.test.t{uuid.uuid4().hex}")
    app.register()
    return app


def _make_window(cfg: Config | None = None) -> MainWindow:
    app = _make_app()
    win = MainWindow(app, cfg or Config())
    app.window = win  # type: ignore[attr-defined]
    return win


def _make_photo_root(tmp_path: Path) -> Path:
    """Create a photo root with numeric YYYY/MM folders and non-numeric siblings."""
    root = tmp_path / "Bilder"
    for folder in ("2004/01", "2004/02", "2005/03", "2024/05"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    # Non-numeric siblings must be ignored by the tree.
    (root / "Christiane").mkdir()
    (root / "2024" / "misc").mkdir(parents=True, exist_ok=True)
    (root / "24").mkdir()
    return root


def _find_widget(widget: Gtk.Widget, cls: type) -> Gtk.Widget | None:
    """Depth-first search for the first descendant of type *cls*."""
    if isinstance(widget, cls):
        return widget
    if hasattr(widget, "get_first_child"):
        child = widget.get_first_child()
        while child is not None:
            found = _find_widget(child, cls)
            if found is not None:
                return found
            child = child.get_next_sibling()
    return None


def _tree_rows(store: Gtk.TreeStore) -> list[tuple[str, str, list[str]]]:
    """Return [(year, kind, [months...]), ...] for the nav store."""
    rows: list[tuple[str, str, list[str]]] = []
    for i in range(store.iter_n_children(None)):
        year_iter = store.iter_nth_child(None, i)
        year = store.get_value(year_iter, 0)
        kind = store.get_value(year_iter, 1)
        months = [
            store.get_value(store.iter_nth_child(year_iter, j), 0)
            for j in range(store.iter_n_children(year_iter))
        ]
        rows.append((year, kind, months))
    return rows


class TestLayoutRegions:
    def test_window_constructs_with_all_regions(self):
        win = _make_window()
        # Toolbar controls
        assert win.mode_selector is not None
        assert win.scan_button is not None
        assert win.select_all_button is not None
        assert win.select_none_button is not None
        assert win.add_to_queue_button is not None
        assert win.execute_queue_button is not None
        assert win.discard_queue_button is not None
        # Left pane tree
        assert win.nav_tree is not None
        assert win.nav_store is not None
        # Right 2x4 grid
        assert win.thumb_grid is not None
        assert len(win._grid_cells) == 8
        # Bottom tabs: Queue and Log
        assert win.notebook.get_n_pages() == 2
        # Status bar
        assert win.status_label is not None

    def test_scan_button_starts_disabled(self):
        win = _make_window()
        assert win.scan_button.get_sensitive() is False

    def test_log_panel_has_initial_message(self):
        win = _make_window()
        text = win.log_buffer.get_text(win.log_buffer.get_start_iter(), win.log_buffer.get_end_iter(), False)
        assert "ready" in text.lower()

    def test_grid_is_two_rows_four_columns(self):
        win = _make_window()
        grid = win.thumb_grid
        layout = grid.get_layout_manager()
        # 8 cells attached at (col, row) with col in 0..3 and row in 0..1.
        positions = set()
        for cell in win._grid_cells:
            child = layout.get_layout_child(cell)
            positions.add((child.get_property("row"), child.get_property("column")))
        assert positions == {(r, c) for r in range(2) for c in range(4)}


class TestModeSelector:
    def test_default_mode_is_similarity(self):
        win = _make_window()
        assert win.mode == "similarity"
        assert win.mode_selector.get_selected() == 0

    def test_mode_selector_has_exactly_two_states(self):
        win = _make_window()
        model = win.mode_selector.get_model()
        assert model.get_n_items() == 2
        assert model.get_string(0) == "Similarity"
        assert model.get_string(1) == "Blur"

    def test_switching_to_blur_and_back(self):
        win = _make_window()
        win.mode_selector.set_selected(1)
        assert win.mode == "blur"
        win.mode_selector.set_selected(0)
        assert win.mode == "similarity"


class TestYearMonthTree:
    def test_tree_populated_from_photo_root(self, tmp_path):
        root = _make_photo_root(tmp_path)
        cfg = Config(photo_root=str(root))
        win = _make_window(cfg)
        rows = _tree_rows(win.nav_store)
        assert rows == [
            ("2004", "year", ["01", "02"]),
            ("2005", "year", ["03"]),
            ("2024", "year", ["05"]),
        ]

    def test_non_numeric_folders_are_absent(self, tmp_path):
        root = _make_photo_root(tmp_path)
        cfg = Config(photo_root=str(root))
        win = _make_window(cfg)
        rows = _tree_rows(win.nav_store)
        labels = [year for year, _, _ in rows]
        assert "Christiane" not in labels
        assert "24" not in labels
        months = [m for _, _, months in rows for m in months]
        assert "misc" not in months

    def test_empty_archive_yields_empty_tree(self, tmp_path):
        root = tmp_path / "Bilder"
        root.mkdir()
        cfg = Config(photo_root=str(root))
        win = _make_window(cfg)
        assert _tree_rows(win.nav_store) == []

    def test_missing_photo_root_yields_empty_tree(self, tmp_path):
        cfg = Config(photo_root=str(tmp_path / "nope"))
        win = _make_window(cfg)
        assert _tree_rows(win.nav_store) == []

    def test_reload_is_idempotent(self, tmp_path):
        root = _make_photo_root(tmp_path)
        cfg = Config(photo_root=str(root))
        win = _make_window(cfg)
        win.reload_year_months()
        assert _tree_rows(win.nav_store) == [
            ("2004", "year", ["01", "02"]),
            ("2005", "year", ["03"]),
            ("2024", "year", ["05"]),
        ]


class TestScanButtonWiring:
    def test_selecting_year_does_not_enable_scan(self, tmp_path):
        root = _make_photo_root(tmp_path)
        cfg = Config(photo_root=str(root))
        win = _make_window(cfg)
        year_iter = win.nav_store.iter_nth_child(None, 0)
        win.nav_selection.select_iter(year_iter)
        assert win.scan_button.get_sensitive() is False

    def test_selecting_month_enables_scan(self, tmp_path):
        root = _make_photo_root(tmp_path)
        cfg = Config(photo_root=str(root))
        win = _make_window(cfg)
        year_iter = win.nav_store.iter_nth_child(None, 0)
        month_iter = win.nav_store.iter_nth_child(year_iter, 0)
        win.nav_selection.select_iter(month_iter)
        assert win.scan_button.get_sensitive() is True

    def test_selected_month_returns_year_and_month(self, tmp_path):
        root = _make_photo_root(tmp_path)
        cfg = Config(photo_root=str(root))
        win = _make_window(cfg)
        year_iter = win.nav_store.iter_nth_child(None, 0)
        month_iter = win.nav_store.iter_nth_child(year_iter, 0)
        win.nav_selection.select_iter(month_iter)
        assert win.selected_month() == ("2004", "01")

    def test_no_selection_returns_none(self, tmp_path):
        root = _make_photo_root(tmp_path)
        cfg = Config(photo_root=str(root))
        win = _make_window(cfg)
        assert win.selected_month() is None


@pytest.mark.skipif(not HAS_DISPLAY, reason="No GTK display available")
class TestResizableLayout:
    def test_regions_remain_visible_at_small_size(self):
        win = _make_window()
        win.set_default_size(1024, 768)
        win.present()
        _run_until_idle(win)
        paned = _find_widget(win, Gtk.Paned)
        assert paned is not None
        assert paned.get_allocation().width > 0
        assert paned.get_allocation().height > 0
        assert win.notebook.get_allocation().height > 0
        assert win.status_label.get_allocation().height > 0
        start = paned.get_start_child()
        end = paned.get_end_child()
        assert start.get_allocation().width > 0
        assert end.get_allocation().width > 0
        win.close()

    def test_regions_remain_visible_at_large_size(self):
        win = _make_window()
        win.set_default_size(1600, 1000)
        win.present()
        _run_until_idle(win)
        paned = _find_widget(win, Gtk.Paned)
        assert paned is not None
        assert paned.get_allocation().width > 0
        assert paned.get_allocation().height > 0
        assert win.notebook.get_allocation().height > 0
        assert win.status_label.get_allocation().height > 0
        win.close()


def _run_until_idle(win: MainWindow) -> None:
    """Run the GTK main loop until the window has been allocated once."""
    import gi

    gi.require_version("GLib", "2.0")
    from gi.repository import GLib

    loop = GLib.MainLoop()

    def stop() -> bool:
        loop.quit()
        return False

    GLib.timeout_add(500, stop)
    loop.run()
