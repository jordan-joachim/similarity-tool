"""GTK4 main window for the Similarity Tool.

This module provides the application shell: a toolbar with the mode selector
(Similarity / Blur), a left navigation/result pane, a right thumbnail grid
area, and a bottom tabbed area with Queue and Log panels. Detection features
are wired in by later milestones; the skeleton focuses on launching cleanly
and rendering the layout regions.
"""

from __future__ import annotations

import logging
import os

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, Gtk  # noqa: E402

from similarity_tool import __version__  # noqa: E402
from similarity_tool.config import Config, ensure_config_file, load_config  # noqa: E402
from similarity_tool.scanner import list_year_months

log = logging.getLogger(__name__)

APP_ID = "io.github.joachim.similaritytool"


class _MessageHandler(logging.Handler):
    """Collect log records into a list so they can be shown in the Log panel."""

    def __init__(self, messages: list[str]) -> None:
        super().__init__()
        self.messages = messages

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        self.messages.append(self.format(record))


class SimilarityToolApplication(Gtk.Application):
    """The Similarity Tool GTK application."""

    def __init__(self) -> None:
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.window: MainWindow | None = None
        self.config: Config | None = None

    def do_startup(self) -> None:  # noqa: D102
        Gtk.Application.do_startup(self)

    def do_activate(self) -> None:  # noqa: D102
        if self.window is None:
            self.config, config_messages = self._load_settings()
            self.window = MainWindow(self, self.config)
            for message in config_messages:
                self.window._log_message(message)
            self.window.present()
        else:
            self.window.present()

    def _load_settings(self) -> tuple[Config, list[str]]:
        """Load or create config, ensuring the config and cache directories exist.

        Returns the config and any messages (e.g. malformed-config errors) that
        should be surfaced in the Log panel.
        """
        messages: list[str] = []
        try:
            from similarity_tool import config as config_mod

            handler = _MessageHandler(messages)
            config_mod.log.addHandler(handler)
            try:
                ensure_config_file()
                cfg = load_config()
                config_mod.create_cache_dir(cfg)
            finally:
                config_mod.log.removeHandler(handler)
            log.info("Using config at %s", os.path.expanduser("~/.config/similarity-tool/config.json"))
            return cfg, messages
        except Exception as exc:  # noqa: BLE001 - config must never prevent launch
            log.error("Could not initialize configuration (%s); using defaults.", exc)
            messages.append(f"Error: could not initialize configuration ({exc}); using defaults.")
            return Config(), messages


class MainWindow(Gtk.ApplicationWindow):
    """The main window with the four layout regions."""

    def __init__(self, app: SimilarityToolApplication, cfg: Config) -> None:
        super().__init__(application=app)
        self.app = app
        self.cfg = cfg
        self.mode: str = "similarity"

        self.set_title(f"Similarity Tool {__version__}")
        self.set_default_size(1200, 800)
        # Keep all four regions usable when the window is shrunk: the toolbar,
        # left pane, right grid, and bottom tabs each keep a minimum size.
        self.set_size_request(800, 600)

        self._build_ui()
        self.reload_year_months()
        self._log_message(f"Similarity Tool {__version__} ready")

    def _build_ui(self) -> None:
        # Root vertical box: toolbar / content / status bar.
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_child(root)

        root.append(self._build_toolbar())

        # Content: horizontal paned with left pane and right grid.
        content = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        content.set_wide_handle(True)
        content.set_resize_start_child(False)
        content.set_resize_end_child(True)
        content.set_shrink_start_child(False)
        content.set_shrink_end_child(False)
        root.append(content)

        self.nav_tree = self._build_nav_tree()
        left_scroll = Gtk.ScrolledWindow()
        left_scroll.set_child(self.nav_tree)
        left_scroll.set_min_content_width(220)
        left_scroll.set_vexpand(True)
        content.set_start_child(left_scroll)

        self.grid_area = self._build_grid_area()
        right_scroll = Gtk.ScrolledWindow()
        right_scroll.set_child(self.grid_area)
        right_scroll.set_hexpand(True)
        right_scroll.set_vexpand(True)
        content.set_end_child(right_scroll)
        content.set_position(280)

        # Bottom: tabs for Queue and Log.
        self.notebook = Gtk.Notebook()
        self.notebook.set_vexpand(False)
        self.notebook.set_size_request(-1, 180)
        root.append(self.notebook)

        self.queue_placeholder = self._build_queue_page()
        self.notebook.append_page(self.queue_placeholder, Gtk.Label(label="Queue"))

        self.log_view = Gtk.TextView()
        self.log_view.set_editable(False)
        self.log_view.set_cursor_visible(False)
        self.log_buffer = self.log_view.get_buffer()
        log_scroll = Gtk.ScrolledWindow()
        log_scroll.set_child(self.log_view)
        log_scroll.set_size_request(-1, 160)
        self.notebook.append_page(log_scroll, Gtk.Label(label="Log"))

        self.status_label = Gtk.Label(label="Ready")
        self.status_label.set_halign(Gtk.Align.START)
        self.status_label.set_margin_top(2)
        self.status_label.set_margin_bottom(2)
        self.status_label.set_margin_start(6)
        root.append(self.status_label)

        self._add_shortcuts()

    def _build_toolbar(self) -> Gtk.Box:
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_margin_top(6)
        toolbar.set_margin_bottom(6)
        toolbar.set_margin_start(6)
        toolbar.set_margin_end(6)

        mode_label = Gtk.Label(label="Mode:")
        toolbar.append(mode_label)

        self.mode_selector = Gtk.DropDown.new_from_strings(["Similarity", "Blur"])
        self.mode_selector.connect("notify::selected", self._on_mode_changed)
        toolbar.append(self.mode_selector)

        separator = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        toolbar.append(separator)

        self.scan_button = Gtk.Button(label="Scan")
        self.scan_button.set_sensitive(False)
        toolbar.append(self.scan_button)

        self.select_all_button = Gtk.Button(label="Select All")
        self.select_all_button.set_sensitive(False)
        toolbar.append(self.select_all_button)

        self.select_none_button = Gtk.Button(label="Select None")
        self.select_none_button.set_sensitive(False)
        toolbar.append(self.select_none_button)

        separator2 = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        toolbar.append(separator2)

        self.add_to_queue_button = Gtk.Button(label="Add to Queue")
        self.add_to_queue_button.set_sensitive(False)
        toolbar.append(self.add_to_queue_button)

        self.execute_queue_button = Gtk.Button(label="Execute Queue")
        self.execute_queue_button.set_sensitive(False)
        toolbar.append(self.execute_queue_button)

        self.discard_queue_button = Gtk.Button(label="Discard Queue")
        self.discard_queue_button.set_sensitive(False)
        toolbar.append(self.discard_queue_button)

        return toolbar

    def _build_nav_tree(self) -> Gtk.TreeView:
        self.nav_store = Gtk.TreeStore.new([str, str])  # (label, node kind)
        tree = Gtk.TreeView(model=self.nav_store)
        renderer = Gtk.CellRendererText()
        column = Gtk.TreeViewColumn(title="Year / Month", cell_renderer=renderer, text=0)
        tree.append_column(column)
        tree.set_headers_visible(False)
        self.nav_selection = tree.get_selection()
        self.nav_selection.connect("changed", self._on_nav_selection_changed)
        return tree

    def reload_year_months(self) -> None:
        """(Re)build the year/month tree from the configured photo root.

        Only numeric ``YYYY``/``MM`` folders are shown; non-numeric siblings
        are ignored. The tree is empty when the root is missing or has no
        numeric year folders.
        """
        self.nav_store.clear()
        year_iters: dict[str, Gtk.TreeIter] = {}
        for year, month in list_year_months(self.cfg.photo_root):
            year_iter = year_iters.get(year)
            if year_iter is None:
                year_iter = self.nav_store.append(None, [year, "year"])
                year_iters[year] = year_iter
            self.nav_store.append(year_iter, [month, "month"])
        # Expand all years so months are visible and selectable at launch;
        # clicking a year node still collapses/expands its months.
        self.nav_tree.expand_all()
        self._on_nav_selection_changed(self.nav_selection)

    def selected_month(self) -> tuple[str, str] | None:
        """Return the ``(year, month)`` of the selected tree node, or ``None``.

        Only a month node (a child of a year node) is a valid scan target;
        selecting a year node returns ``None`` so the Scan button stays
        disabled.
        """
        model, tree_iter = self.nav_selection.get_selected()
        if tree_iter is None:
            return None
        parent = model.iter_parent(tree_iter)
        if parent is None:
            return None
        return model.get_value(parent, 0), model.get_value(tree_iter, 0)

    def _on_nav_selection_changed(self, _selection: Gtk.TreeSelection) -> None:
        """Enable Scan only when a month node is selected."""
        self.scan_button.set_sensitive(self.selected_month() is not None)

    def _build_grid_area(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        box.set_margin_start(6)
        box.set_margin_end(6)

        self.grid_header = Gtk.Label(label="Select a month and press Scan")
        self.grid_header.set_halign(Gtk.Align.START)
        box.append(self.grid_header)

        grid = Gtk.Grid()
        grid.set_row_spacing(6)
        grid.set_column_spacing(6)
        grid.set_halign(Gtk.Align.FILL)
        grid.set_valign(Gtk.Align.FILL)
        self.thumb_grid = grid

        # 2x4 placeholder cells so the layout regions are visible at launch.
        self._grid_cells: list[Gtk.Box] = []
        for index in range(8):
            cell = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            cell.set_hexpand(True)
            cell.set_vexpand(True)
            placeholder = Gtk.Label(label=f"Cell {index + 1}")
            placeholder.set_hexpand(True)
            placeholder.set_vexpand(True)
            placeholder.set_valign(Gtk.Align.CENTER)
            cell.append(placeholder)
            row, col = divmod(index, 4)
            grid.attach(cell, col, row, 1, 1)
            self._grid_cells.append(cell)

        box.append(grid)
        return box

    def _build_queue_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        box.set_margin_start(6)
        box.set_margin_end(6)
        self.queue_count_label = Gtk.Label(label="Queue: 0 items (0 bytes)")
        self.queue_count_label.set_halign(Gtk.Align.START)
        box.append(self.queue_count_label)
        empty_label = Gtk.Label(label="No items queued.")
        empty_label.set_halign(Gtk.Align.START)
        box.append(empty_label)
        return box

    def _add_shortcuts(self) -> None:
        def escape_cb(_widget: Gtk.Widget, _state: int) -> bool:
            self.close()
            return True

        controller = Gtk.EventControllerKey.new()
        controller.connect("key-pressed", escape_cb)
        self.add_controller(controller)

    def _on_mode_changed(self, dropdown: Gtk.DropDown, _param: object) -> None:
        selected = dropdown.get_selected()
        if selected == 0:
            self.mode = "similarity"
        elif selected == 1:
            self.mode = "blur"
        self._log_message(f"Mode switched to {self.mode}")

    def _log_message(self, message: str) -> None:
        """Append a line to the log panel."""
        if hasattr(self, "log_buffer"):
            self.log_buffer.insert_at_cursor(f"{message}\n")


def main() -> int:
    """Entry point used by tests and the console script."""
    logging.basicConfig(level=logging.INFO)
    app = SimilarityToolApplication()
    return app.run(None)


if __name__ == "__main__":
    raise SystemExit(main())
