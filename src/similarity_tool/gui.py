"""GTK4 main window for the Similarity Tool.

This module provides the application shell: a toolbar with the mode selector
(Similarity / Blur), a left navigation/result pane, a right thumbnail grid
area, and a bottom tabbed area with Queue and Log panels. The Scan button
starts a similarity or blur scan for the selected month on a background
thread, shows a progress indicator, keeps the GTK main loop responsive, and
populates the left result list and right grid when the scan finishes. Empty
months and invalid month paths are surfaced as informative messages instead of
starting a long no-op, and switching mode during an active scan cancels the
in-progress work without crashing.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, GLib, Gtk  # noqa: E402

from similarity_tool import __version__
from similarity_tool.ai_refinement import cluster_label, refine_clusters
from similarity_tool.blur import scan_blur
from similarity_tool.clusters import build_clusters
from similarity_tool.config import Config, ensure_config_file, load_config
from similarity_tool.hashing import HashCache
from similarity_tool.models import BlurCandidate, Cluster
from similarity_tool.scanner import list_year_months, scan_month

log = logging.getLogger(__name__)

APP_ID = "io.github.joachim.similaritytool"


@dataclass
class _ScanOutcome:
    """The result of a background scan, marshalled back to the main thread.

    ``mode`` is the detection mode the scan ran in, ``result`` holds the
    clusters or blur candidates (empty when none were found), ``empty_month``
    is True when the month contained no supported images at all, and ``error``
    carries a failure message when the scan raised.
    """

    mode: str
    result: list[Cluster] | list[BlurCandidate] = field(default_factory=list)
    empty_month: bool = False
    error: str | None = None


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

        # Scan state. Scans run on a background thread; ``_scanning`` is the
        # single source of truth for whether a scan is in progress, and
        # ``_scan_generation`` lets a mode switch or month change invalidate a
        # scan that is still running.
        self._scanning: bool = False
        self._scan_thread: threading.Thread | None = None
        self._scan_generation: int = 0

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
        left_pane = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        left_pane.append(self.nav_tree)
        left_pane.append(self._build_result_list())
        left_scroll = Gtk.ScrolledWindow()
        left_scroll.set_child(left_pane)
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
        self.scan_button.connect("clicked", self._on_scan_clicked)
        toolbar.append(self.scan_button)

        # Progress indicator: a spinner plus a status label. The spinner is
        # hidden until a scan starts and stops when the scan finishes.
        self.scan_spinner = Gtk.Spinner()
        self.scan_spinner.set_visible(False)
        toolbar.append(self.scan_spinner)
        self.scan_status_label = Gtk.Label(label="")
        toolbar.append(self.scan_status_label)

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

    def _build_result_list(self) -> Gtk.Widget:
        """Build the left-pane result list below the year/month tree.

        The list shows one row per cluster (Similarity mode) or per blur
        candidate (Blur mode). Selecting a row loads its images into the right
        grid. The header doubles as the empty-state label.
        """
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        box.set_margin_start(6)
        box.set_margin_end(6)

        self.result_header = Gtk.Label(label="Select a month and press Scan")
        self.result_header.set_halign(Gtk.Align.START)
        self.result_header.set_wrap(True)
        box.append(self.result_header)

        self.result_list = Gtk.ListBox()
        self.result_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.result_list.connect("row-selected", self._on_result_row_selected)
        scroll = Gtk.ScrolledWindow()
        scroll.set_child(self.result_list)
        scroll.set_vexpand(True)
        box.append(scroll)
        return box

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
        """Enable Scan only when a month node is selected.

        Selecting a different month also cancels any in-progress scan and
        resets the current results so the grid never shows stale images for
        the previously selected month.
        """
        if self._scanning:
            self._cancel_scan("month changed")
        self.scan_button.set_sensitive(self.selected_month() is not None)
        self._clear_results()

    def _cancel_scan(self, reason: str) -> None:
        """Cancel the in-progress scan and reset the UI to idle.

        The running worker thread is daemon and keeps executing, but its
        result is discarded: the generation is bumped so the worker's
        ``_finish_scan`` callback is treated as stale.
        """
        self._scan_generation += 1
        self._scanning = False
        self._scan_thread = None
        self.scan_spinner.stop()
        self.scan_spinner.set_visible(False)
        self.scan_status_label.set_text("")
        self.scan_button.set_sensitive(self.selected_month() is not None)
        self._log_message(f"Scan cancelled ({reason}).")

    def _clear_results(self) -> None:
        """Clear the result list, grid, and empty-state header.

        Called when the month changes, when a scan starts, and when a scan is
        cancelled. The grid cells are reset to their placeholder labels.
        """
        self.result_list.remove_all()
        self.result_header.set_text("Select a month and press Scan")
        for cell in self._grid_cells:
            child = cell.get_first_child()
            if isinstance(child, Gtk.Label):
                child.set_text("")

    def _on_scan_clicked(self, _button: Gtk.Button) -> None:
        """Start a scan for the selected month, or surface an invalid path.

        The month folder is validated on the main thread first: a missing or
        empty folder is reported immediately without starting a background
        scan. Otherwise the scan runs on a worker thread so the GTK main loop
        stays responsive, and the result is marshalled back with
        ``GLib.idle_add``.
        """
        if self._scanning:
            return
        selected = self.selected_month()
        if selected is None:
            self._log_message("No month selected; nothing to scan.")
            return
        year, month = selected
        month_dir = Path(self.cfg.photo_root) / year / month
        if not month_dir.is_dir():
            message = f"Folder {year}/{month} does not exist; nothing to scan."
            self._log_message(message)
            self.result_header.set_text(message)
            return

        self._scanning = True
        self._scan_generation += 1
        generation = self._scan_generation
        mode = self.mode
        self.scan_button.set_sensitive(False)
        self._clear_results()
        self.scan_spinner.set_visible(True)
        self.scan_spinner.start()
        self.scan_status_label.set_text(f"Scanning {year}/{month} ({mode})...")
        self._log_message(f"Scanning {year}/{month} ({mode})...")

        self._scan_thread = threading.Thread(
            target=self._scan_worker,
            args=(year, month, mode, generation),
            daemon=True,
        )
        self._scan_thread.start()

    def _scan_worker(self, year: str, month: str, mode: str, generation: int) -> None:
        """Run the scan off the main thread and marshal the result back.

        The result is delivered with ``GLib.idle_add`` only when *generation*
        still matches the current scan generation; a mode switch or month
        change bumps the generation, so a stale scan's result is discarded
        instead of being applied to the new UI state.
        """
        try:
            if mode == "similarity":
                outcome = self._run_similarity_scan(year, month)
            else:
                outcome = self._run_blur_scan(year, month)
        except Exception as exc:  # a scan failure must never kill the app
            log.exception("Scan failed for %s/%s", year, month)
            outcome = _ScanOutcome(mode=mode, error=f"Scan failed: {exc}")
        GLib.idle_add(self._finish_scan, generation, outcome)

    def _run_similarity_scan(self, year: str, month: str) -> _ScanOutcome:
        """Hash and cluster the selected month's images (Similarity mode)."""
        photos = scan_month(
            self.cfg.photo_root, year, month, self.cfg.file_extensions
        )
        if not photos:
            return _ScanOutcome(mode="similarity", empty_month=True)
        cache = HashCache(self.cfg.resolved_cache_path(), algorithms=self.cfg.hash_algorithms)
        try:
            records = cache.compute_hashes(photos)
        finally:
            cache.close()
        result = build_clusters(
            records,
            photos,
            phash_threshold=self.cfg.phash_threshold,
            dhash_threshold=self.cfg.dhash_threshold,
            hash_algorithms=self.cfg.hash_algorithms,
        )
        clusters = refine_clusters(result.clusters, self.cfg)
        return _ScanOutcome(mode="similarity", result=clusters)

    def _run_blur_scan(self, year: str, month: str) -> _ScanOutcome:
        """Score the selected month's images and return blur candidates."""
        photos = scan_month(
            self.cfg.photo_root, year, month, self.cfg.file_extensions
        )
        if not photos:
            return _ScanOutcome(mode="blur", empty_month=True)
        result = scan_blur(photos, self.cfg)
        return _ScanOutcome(mode="blur", result=result.candidates)

    def _finish_scan(self, generation: int, outcome: _ScanOutcome) -> bool:
        """Apply a finished scan's result to the UI (runs on the main thread).

        Returns ``False`` so the idle callback runs only once. A stale result
        (generation mismatch) is discarded. The progress indicator is always
        stopped and the Scan button re-enabled.
        """
        if generation != self._scan_generation:
            # A newer scan (or a mode switch / month change) superseded this
            # one; the cancelling handler already reset the UI state, so this
            # stale callback must not touch it.
            return False

        self._scanning = False
        self._scan_thread = None
        self.scan_spinner.stop()
        self.scan_spinner.set_visible(False)
        self.scan_status_label.set_text("")
        self.scan_button.set_sensitive(self.selected_month() is not None)

        if outcome.error is not None:
            self._log_message(outcome.error)
            self.result_header.set_text(outcome.error)
            return False

        if outcome.empty_month:
            message = "No images found in this month"
            self._log_message(message)
            self.result_header.set_text(message)
            return False

        if outcome.mode == "similarity":
            self._populate_clusters(outcome.result or [])
        else:
            self._populate_candidates(outcome.result or [])
        self._log_message("Scan finished.")
        return False

    def _populate_clusters(self, clusters: list[Cluster]) -> None:
        """Populate the result list and grid with similarity clusters."""
        self.result_list.remove_all()
        if not clusters:
            self.result_header.set_text("No similar images found")
            return
        self.result_header.set_text(f"{len(clusters)} cluster(s) found")
        for index, cluster in enumerate(clusters, start=1):
            row = Gtk.ListBoxRow()
            row.title = cluster_label(cluster, index)
            row.result = cluster
            label = Gtk.Label(label=row.title, xalign=0.0)
            row.set_child(label)
            self.result_list.append(row)
        self.result_list.select_row(self.result_list.get_row_at_index(0))

    def _populate_candidates(self, candidates: list[BlurCandidate]) -> None:
        """Populate the result list and grid with blur candidates."""
        self.result_list.remove_all()
        if not candidates:
            self.result_header.set_text("No blurry images found")
            return
        self.result_header.set_text(f"{len(candidates)} candidate(s) found")
        for index, candidate in enumerate(candidates, start=1):
            row = Gtk.ListBoxRow()
            row.title = (
                f"{index}. {candidate.photo.name} "
                f"(score {candidate.score:.1f}, {candidate.percentile:.0f}%)"
            )
            row.result = candidate
            label = Gtk.Label(label=row.title, xalign=0.0)
            row.set_child(label)
            self.result_list.append(row)
        self.result_list.select_row(self.result_list.get_row_at_index(0))

    def _on_result_row_selected(
        self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow | None
    ) -> None:
        """Load the selected result's images into the right-hand grid.

        The grid shows the member images of the selected cluster, or the
        single candidate image in Blur mode. Stale thumbnails from a previous
        selection are cleared first.
        """
        for cell in self._grid_cells:
            child = cell.get_first_child()
            if isinstance(child, Gtk.Label):
                child.set_text("")
        if row is None:
            return
        result = getattr(row, "result", None)
        if isinstance(result, Cluster):
            for cell, photo in zip(self._grid_cells, result.members):
                child = cell.get_first_child()
                if isinstance(child, Gtk.Label):
                    child.set_text(photo.name)
        elif isinstance(result, BlurCandidate):
            child = self._grid_cells[0].get_first_child()
            if isinstance(child, Gtk.Label):
                child.set_text(result.photo.name)

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
        # A mode switch cancels any in-progress scan: bump the generation so
        # the running worker's result is discarded, and reset the UI to idle.
        if self._scanning:
            self._cancel_scan("mode switched")
            self._clear_results()
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
