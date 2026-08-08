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

import io
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, Gio, GLib, Gtk, Pango  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from similarity_tool import __version__
from similarity_tool.ai_refinement import cluster_label, refine_clusters
from similarity_tool.blur import scan_blur
from similarity_tool.clusters import build_clusters
from similarity_tool.config import Config, ensure_config_file, load_config
from similarity_tool.hashing import HashCache
from similarity_tool.models import BlurCandidate, Cluster, QueueItem
from similarity_tool.scanner import list_year_months, scan_month
from similarity_tool.trash import TrashFailure, TrashResult, move_to_trash

log = logging.getLogger(__name__)

APP_ID = "io.github.joachim.similaritytool"

# Maximum edge of a downscaled thumbnail. The full-resolution image is never
# loaded into the grid; previews are generated on a background thread and
# scaled to fit this box while preserving the aspect ratio.
THUMB_MAX = 200


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


def _human_size(size: int) -> str:
    """Format a byte count in a human-readable form (e.g. ``1.2 MB``)."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _load_thumbnail(path: Path) -> Gdk.Texture | None:
    """Downscale *path* to a thumbnail texture, preserving the aspect ratio.

    The image is decoded with Pillow, downscaled to fit ``THUMB_MAX`` on its
    longest edge, and encoded as PNG bytes for ``Gdk.Texture``. The
    full-resolution image is never loaded into the grid widget. Returns
    ``None`` when the file cannot be decoded.
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((THUMB_MAX, THUMB_MAX), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    return Gdk.Texture.new_from_bytes(GLib.Bytes.new(buffer.getvalue()))


def _click_hits_checkbox(cell: Gtk.Box, x: float, y: float) -> bool:
    """Return True when a click at (*x*, *y*) in *cell* lands on its checkbox.

    Clicks on the checkbox are handled by the checkbox itself; the cell's
    click gesture must not toggle the selection again in that case.
    """
    checkbox = getattr(cell, "checkbox", None)
    if checkbox is None:
        return False
    allocation = checkbox.get_allocation()
    if allocation.width <= 0 or allocation.height <= 0:
        # The widget is not realized (e.g. headless tests); fall back to
        # treating the click as a cell click.
        return False
    return allocation.x <= x <= allocation.x + allocation.width and allocation.y <= y <= allocation.y + allocation.height


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

        # Thumbnail state. Previews are downscaled on a background thread so
        # the main loop stays responsive; ``_thumb_generation`` invalidates
        # stale thumbnail loads when the grid is repopulated.
        self._thumb_thread: threading.Thread | None = None
        self._thumb_generation: int = 0

        # Grid selection state. ``_selection`` holds the indices of the
        # currently selected cells in the 2x4 grid, and ``_selection_by_result``
        # remembers each result's selection so it does not leak across result
        # rows: switching away and back restores the per-result selection.
        # Results are keyed by a monotonic id assigned to each result-list row
        # (never by ``id(result)``, which Python may reuse after a scan).
        self._selection: set[int] = set()
        self._selection_by_result: dict[int, set[int]] = {}
        self._current_result_id: int | None = None
        self._next_result_id: int = 0
        self._focused_cell_index: int = 0

        # Deletion queue state. ``_queue`` holds the staged items in display
        # order; it is the single source of truth for the queue tab. Items are
        # keyed by absolute path so a file can never be staged twice, and the
        # queue survives mode switches and month changes (it is only cleared
        # by Discard Queue, Execute Queue, or a fresh application launch).
        self._queue: list[QueueItem] = []
        self._queue_by_path: dict[str, QueueItem] = {}
        self._execution_thread: threading.Thread | None = None
        self._queue_thumb_thread: threading.Thread | None = None
        self._queue_thumb_generation: int = 0

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

        # Style the queue source-mode badge (Similarity / Blur).
        provider = Gtk.CssProvider()
        provider.load_from_data(
            b".mode-badge { font-size: 10px; opacity: 0.8; }"
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

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

        self.queue_page = self._build_queue_page()
        self.notebook.append_page(self.queue_page, Gtk.Label(label="Queue"))

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
        self.select_all_button.connect("clicked", self._on_select_all_clicked)
        toolbar.append(self.select_all_button)

        self.select_none_button = Gtk.Button(label="Select None")
        self.select_none_button.set_sensitive(False)
        self.select_none_button.connect("clicked", self._on_select_none_clicked)
        toolbar.append(self.select_none_button)

        separator2 = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        toolbar.append(separator2)

        self.add_to_queue_button = Gtk.Button(label="Add to Queue")
        self.add_to_queue_button.set_sensitive(False)
        self.add_to_queue_button.connect("clicked", self._on_add_to_queue_clicked)
        toolbar.append(self.add_to_queue_button)

        self.execute_queue_button = Gtk.Button(label="Execute Queue")
        self.execute_queue_button.set_sensitive(False)
        self.execute_queue_button.connect("clicked", self._on_execute_queue_clicked)
        toolbar.append(self.execute_queue_button)

        self.discard_queue_button = Gtk.Button(label="Discard Queue")
        self.discard_queue_button.set_sensitive(False)
        self.discard_queue_button.connect("clicked", self._on_discard_queue_clicked)
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
        cancelled. The grid cells are reset to their placeholder state.
        """
        self.result_list.remove_all()
        self.result_header.set_text("Select a month and press Scan")
        self._clear_grid()

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
        # All previous result rows are gone, so their per-result selection
        # state is meaningless; drop it to keep memory bounded across scans.
        self._selection_by_result.clear()
        if not clusters:
            self.result_header.set_text("No similar images found")
            return
        self.result_header.set_text(f"{len(clusters)} cluster(s) found")
        for index, cluster in enumerate(clusters, start=1):
            row = Gtk.ListBoxRow()
            row.title = cluster_label(cluster, index)
            row.result = cluster
            row.result_id = self._next_result_id
            self._next_result_id += 1
            label = Gtk.Label(label=row.title, xalign=0.0)
            row.set_child(label)
            self.result_list.append(row)
        self.result_list.select_row(self.result_list.get_row_at_index(0))

    def _populate_candidates(self, candidates: list[BlurCandidate]) -> None:
        """Populate the result list and grid with blur candidates."""
        self.result_list.remove_all()
        # All previous result rows are gone, so their per-result selection
        # state is meaningless; drop it to keep memory bounded across scans.
        self._selection_by_result.clear()
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
            row.result_id = self._next_result_id
            self._next_result_id += 1
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
        selection are cleared first, and the per-result selection state is
        restored so selections never leak across result rows.
        """
        if row is None:
            self._clear_grid()
            return
        result = getattr(row, "result", None)
        result_id = getattr(row, "result_id", None)
        if isinstance(result, Cluster):
            photos = result.members
            tooltips = None
        elif isinstance(result, BlurCandidate):
            photos = [result.photo]
            tooltips = [f"Blur score {result.score:.1f}, {result.percentile:.0f}%"]
        else:
            self._clear_grid()
            return
        self._load_grid(result_id, photos, tooltips=tooltips)

    def _load_grid(
        self,
        result_id: int | None,
        photos: list,
        tooltips: list[str] | None = None,
    ) -> None:
        """Populate the 2x4 grid with *photos* for the result *result_id*.

        Each populated cell shows a downscaled preview (loaded on a background
        thread), the filename, the human-readable file size, and a checkbox.
        Empty cells become placeholders. The selection state for *result_id*
        is restored from ``_selection_by_result`` so it does not leak across
        result rows.
        """
        # Save the current result's selection before switching away.
        if self._current_result_id is not None:
            self._selection_by_result[self._current_result_id] = set(self._selection)

        self._current_result_id = result_id
        # Clamp the restored selection to the populated cells: after an
        # execution removed some members, stale indices from the previous
        # selection must not linger in the selection set.
        self._selection = {
            i for i in self._selection_by_result.get(result_id, set()) if i < len(photos)
        }
        self._focused_cell_index = 0

        # Invalidate any in-flight thumbnail load from a previous grid.
        self._thumb_generation += 1
        generation = self._thumb_generation

        for index, cell in enumerate(self._grid_cells):
            if index < len(photos):
                photo = photos[index]
                cell.photo = photo
                cell.name_label.set_text(photo.name)
                cell.size_label.set_text(_human_size(photo.size))
                cell.checkbox.set_sensitive(True)
                cell.checkbox.set_active(index in self._selection)
                cell.set_tooltip_text(tooltips[index] if tooltips else None)
                cell.set_visible(True)
            else:
                cell.photo = None
                cell.name_label.set_text("")
                cell.size_label.set_text("")
                cell.checkbox.set_sensitive(False)
                cell.checkbox.set_active(False)
                cell.set_tooltip_text(None)
                cell.set_visible(True)

        self._update_selection_buttons()
        self._update_status()

        # Downscale the previews off the main thread and marshal the textures
        # back with GLib.idle_add. A stale load (generation mismatch) is
        # discarded.
        self._thumb_thread = threading.Thread(
            target=self._thumb_worker,
            args=(photos, generation),
            daemon=True,
        )
        self._thumb_thread.start()

    def _thumb_worker(self, photos: list, generation: int) -> None:
        """Downscale *photos* to thumbnails on a background thread.

        Each photo is decoded with Pillow, downscaled to fit ``THUMB_MAX``
        while preserving the aspect ratio, and encoded as PNG bytes. The
        textures are marshalled back to the main thread with ``GLib.idle_add``
        only when *generation* still matches the current grid generation.
        """
        textures: list[Gdk.Texture | None] = []
        for photo in photos:
            try:
                textures.append(_load_thumbnail(photo.path))
            except Exception:  # a bad image must never crash the grid
                log.warning("Could not load thumbnail for %s", photo.path)
                textures.append(None)
        GLib.idle_add(self._apply_thumbnails, textures, generation)

    def _apply_thumbnails(
        self, textures: list[Gdk.Texture | None], generation: int
    ) -> bool:
        """Apply downscaled previews to the grid cells (main thread).

        Returns ``False`` so the idle callback runs only once. A stale load
        (the grid was repopulated since the thumbnails were generated) is
        discarded.
        """
        if generation != self._thumb_generation:
            return False
        for cell, texture in zip(self._grid_cells, textures):
            cell.picture.set_paintable(texture)
        return False

    def _clear_grid(self) -> None:
        """Reset every grid cell to its placeholder state.

        The current result's selection is saved before clearing so it can be
        restored when the same result is selected again.
        """
        if self._current_result_id is not None:
            self._selection_by_result[self._current_result_id] = set(self._selection)
        self._current_result_id = None
        self._selection = set()
        self._focused_cell_index = 0
        self._thumb_generation += 1
        for cell in self._grid_cells:
            cell.photo = None
            cell.picture.set_paintable(None)
            cell.name_label.set_text("")
            cell.size_label.set_text("")
            cell.checkbox.set_sensitive(False)
            cell.checkbox.set_active(False)
            cell.set_tooltip_text(None)
        self._update_selection_buttons()
        self._update_status()

    def _on_cell_clicked(
        self, _gesture: Gtk.GestureClick, _n_press: int, x: float, y: float, cell: Gtk.Box
    ) -> None:
        """Toggle the clicked cell's selection.

        Clicks that land on the cell's checkbox are left to the checkbox
        itself (which toggles the selection through its own handler); every
        other click on the cell toggles the selection directly.
        """
        if cell.photo is None:
            return
        if _click_hits_checkbox(cell, x, y):
            return
        self._toggle_cell(self._grid_cells.index(cell))

    def _on_cell_checkbox_toggled(self, checkbox: Gtk.CheckButton, cell: Gtk.Box) -> None:
        """Keep the selection set in sync with a cell's checkbox."""
        if cell.photo is None:
            return
        index = self._grid_cells.index(cell)
        if checkbox.get_active():
            self._selection.add(index)
        else:
            self._selection.discard(index)
        self._update_selection_buttons()
        self._update_status()

    def _toggle_cell(self, index: int) -> None:
        """Toggle the selection of the cell at *index*."""
        if index < 0 or index >= len(self._grid_cells):
            return
        cell = self._grid_cells[index]
        if cell.photo is None:
            return
        cell.checkbox.set_active(not cell.checkbox.get_active())

    def _on_select_all_clicked(self, _button: Gtk.Button | None) -> None:
        """Select every populated cell in the current grid."""
        for index, cell in enumerate(self._grid_cells):
            if cell.photo is not None:
                cell.checkbox.set_active(True)

    def _on_select_none_clicked(self, _button: Gtk.Button | None) -> None:
        """Clear the selection in the current grid."""
        for cell in self._grid_cells:
            if cell.photo is not None:
                cell.checkbox.set_active(False)

    def _update_selection_buttons(self) -> None:
        """Enable Select All / Select None / Add to Queue per the grid state."""
        populated = any(cell.photo is not None for cell in self._grid_cells)
        self.select_all_button.set_sensitive(populated)
        self.select_none_button.set_sensitive(populated and bool(self._selection))
        self.add_to_queue_button.set_sensitive(populated and bool(self._selection))

    def _update_status(self) -> None:
        """Show the current selection count in the status bar."""
        self.status_label.set_text(f"{len(self._selection)} selected")

    def _on_grid_key_pressed(
        self, _controller: Gtk.EventControllerKey, keyval: int, _keycode: int, _state: int
    ) -> bool:
        """Handle keyboard selection shortcuts on the thumbnail grid.

        ``Space`` toggles the focused cell, ``A`` selects all populated cells,
        ``N`` clears the selection, and the arrow keys move the focus between
        cells. Returns ``True`` when the key was handled.
        """
        if keyval in (Gdk.KEY_space, Gdk.KEY_Left, Gdk.KEY_Right, Gdk.KEY_Up, Gdk.KEY_Down):
            if self._focused_cell_index >= len(self._grid_cells):
                self._focused_cell_index = 0
            if keyval == Gdk.KEY_space:
                # When a cell's checkbox itself has focus, its own key binding
                # already toggles it; the grid must not toggle a second time.
                if self._focused_widget_is_cell_checkbox():
                    return False
                self._toggle_cell(self._focused_cell_index)
            else:
                self._move_focus(keyval)
            return True
        if keyval in (Gdk.KEY_a, Gdk.KEY_A):
            self._on_select_all_clicked(None)
            return True
        if keyval in (Gdk.KEY_n, Gdk.KEY_N):
            self._on_select_none_clicked(None)
            return True
        return False

    def _focused_widget_is_cell_checkbox(self) -> bool:
        """Return True when the focused widget is a checkbox of a grid cell."""
        focused = self.get_focus()
        if focused is None:
            return False
        for cell in self._grid_cells:
            if focused is cell.checkbox:
                return True
        return False

    def _move_focus(self, keyval: int) -> None:
        """Move the focused cell in the direction of *keyval* (arrow keys)."""
        index = self._focused_cell_index
        row, col = divmod(index, 4)
        if keyval == Gdk.KEY_Left:
            col = max(0, col - 1)
        elif keyval == Gdk.KEY_Right:
            col = min(3, col + 1)
        elif keyval == Gdk.KEY_Up:
            row = max(0, row - 1)
        elif keyval == Gdk.KEY_Down:
            row = min(1, row + 1)
        self._focused_cell_index = row * 4 + col
        # Move the real GTK focus to the newly focused cell so the user can
        # see where the keyboard selection applies.
        self._grid_cells[self._focused_cell_index].grab_focus()

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

        # 2x4 cells. Each cell shows a downscaled preview, the filename, the
        # file size, and a checkbox. Cells are built once and reused; empty
        # cells render as placeholders.
        self._grid_cells: list[Gtk.Box] = []
        for index in range(8):
            cell = self._build_grid_cell()
            row, col = divmod(index, 4)
            grid.attach(cell, col, row, 1, 1)
            self._grid_cells.append(cell)

        # Keyboard selection: Space toggles the focused cell, A selects all,
        # N selects none, arrows move the focus. The controller is attached to
        # the grid so it receives keys while the grid (or a cell inside it)
        # has focus.
        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self._on_grid_key_pressed)
        grid.add_controller(key_controller)

        box.append(grid)
        return box

    def _build_grid_cell(self) -> Gtk.Box:
        """Build one 2x4 grid cell with preview, filename, size, and checkbox.

        The cell carries Python attributes used by the grid logic: ``photo``
        (the PhotoFile shown, or ``None`` for an empty cell), ``picture``,
        ``name_label``, ``size_label``, ``checkbox``, and ``gesture``.
        """
        cell = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        cell.set_hexpand(True)
        cell.set_vexpand(True)
        cell.set_focusable(True)
        cell.set_can_focus(True)
        cell.photo = None  # type: ignore[attr-defined]

        picture = Gtk.Picture()
        picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        picture.set_size_request(THUMB_MAX, THUMB_MAX)
        picture.set_hexpand(True)
        picture.set_vexpand(True)
        cell.append(picture)
        cell.picture = picture  # type: ignore[attr-defined]

        name_label = Gtk.Label(label="")
        name_label.set_ellipsize(Pango.EllipsizeMode.END)
        name_label.set_max_width_chars(24)
        name_label.set_halign(Gtk.Align.CENTER)
        cell.append(name_label)
        cell.name_label = name_label  # type: ignore[attr-defined]

        size_label = Gtk.Label(label="")
        size_label.set_halign(Gtk.Align.CENTER)
        cell.append(size_label)
        cell.size_label = size_label  # type: ignore[attr-defined]

        checkbox = Gtk.CheckButton()
        checkbox.set_halign(Gtk.Align.CENTER)
        checkbox.set_sensitive(False)
        checkbox.connect("toggled", self._on_cell_checkbox_toggled, cell)
        cell.append(checkbox)
        cell.checkbox = checkbox  # type: ignore[attr-defined]

        gesture = Gtk.GestureClick.new()
        gesture.connect("pressed", self._on_cell_clicked, cell)
        cell.add_controller(gesture)
        cell.gesture = gesture  # type: ignore[attr-defined]

        return cell

    def _build_queue_page(self) -> Gtk.Widget:
        """Build the Queue tab: header, thumbnail flow, and empty-state label.

        The header shows the aggregate count and total staged size. The flow
        box renders one thumbnail card per staged item (preview, filename,
        size, source-mode badge, and a remove button). The empty-state label
        is shown when nothing is staged.
        """
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_margin_top(6)
        box.set_margin_bottom(6)
        box.set_margin_start(6)
        box.set_margin_end(6)

        self.queue_count_label = Gtk.Label(label="Queue: 0 items (0 B)")
        self.queue_count_label.set_halign(Gtk.Align.START)
        box.append(self.queue_count_label)

        self.queue_flow = Gtk.FlowBox()
        self.queue_flow.set_max_children_per_line(6)
        self.queue_flow.set_selection_mode(Gtk.SelectionMode.NONE)
        self.queue_flow.set_vexpand(True)
        scroll = Gtk.ScrolledWindow()
        scroll.set_child(self.queue_flow)
        scroll.set_vexpand(True)
        box.append(scroll)

        self.queue_empty_label = Gtk.Label(label="No items queued.")
        self.queue_empty_label.set_halign(Gtk.Align.START)
        box.append(self.queue_empty_label)
        return box

    def _build_queue_card(self, item: QueueItem) -> Gtk.FlowBoxChild:
        """Build one queue thumbnail card for *item*.

        The card shows a downscaled preview, the filename, the human-readable
        file size, a source-mode badge (``Similarity`` or ``Blur``), and a
        remove button. The card carries Python attributes used by the queue
        logic: ``item``, ``picture``, ``name_label``, ``size_label``,
        ``mode_label``, and ``remove_button``.
        """
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        card.set_hexpand(True)
        card.set_vexpand(True)

        picture = Gtk.Picture()
        picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        picture.set_size_request(THUMB_MAX, THUMB_MAX)
        picture.set_hexpand(True)
        picture.set_vexpand(True)
        card.append(picture)

        name_label = Gtk.Label(label=item.photo.name)
        name_label.set_ellipsize(Pango.EllipsizeMode.END)
        name_label.set_max_width_chars(24)
        name_label.set_halign(Gtk.Align.CENTER)
        card.append(name_label)

        size_label = Gtk.Label(label=_human_size(item.photo.size))
        size_label.set_halign(Gtk.Align.CENTER)
        card.append(size_label)

        mode_label = Gtk.Label(label="Similarity" if item.mode == "similarity" else "Blur")
        mode_label.set_halign(Gtk.Align.CENTER)
        mode_label.add_css_class("mode-badge")
        card.append(mode_label)

        remove_button = Gtk.Button(label="Remove")
        remove_button.set_halign(Gtk.Align.CENTER)
        remove_button.connect("clicked", self._on_queue_remove_clicked)
        card.append(remove_button)

        flow_child = Gtk.FlowBoxChild()
        flow_child.set_child(card)
        flow_child.item = item  # type: ignore[attr-defined]
        flow_child.picture = picture  # type: ignore[attr-defined]
        flow_child.name_label = name_label  # type: ignore[attr-defined]
        flow_child.size_label = size_label  # type: ignore[attr-defined]
        flow_child.mode_label = mode_label  # type: ignore[attr-defined]
        flow_child.remove_button = remove_button  # type: ignore[attr-defined]
        return flow_child

    def _on_add_to_queue_clicked(self, _button: Gtk.Button | None) -> None:
        """Stage the currently selected grid photos in the shared queue.

        Only populated cells that are selected are staged. A file already in
        the queue is never staged twice. After staging, the grid selection is
        cleared so the user can keep reviewing without re-offering the same
        images.
        """
        added = 0
        for index in sorted(self._selection):
            cell = self._grid_cells[index]
            if cell.photo is None:
                continue
            item = QueueItem(photo=cell.photo, mode=self.mode)
            if str(item.photo.path) in self._queue_by_path:
                continue
            self._queue.append(item)
            self._queue_by_path[str(item.photo.path)] = item
            self.queue_flow.append(self._build_queue_card(item))
            added += 1
        if added:
            self._log_message(f"Added {added} image(s) to the queue.")
        self._clear_selection()
        self._update_queue_ui()
        if added:
            self._load_queue_thumbnails()

    def _load_queue_thumbnails(self) -> None:
        """Downscale the queue previews on a background thread.

        The textures are marshalled back with ``GLib.idle_add`` only when the
        queue generation still matches, so a stale load (queue rebuilt) is
        discarded.
        """
        self._queue_thumb_generation += 1
        generation = self._queue_thumb_generation
        photos = [item.photo for item in self._queue]
        self._queue_thumb_thread = threading.Thread(
            target=self._queue_thumb_worker,
            args=(photos, generation),
            daemon=True,
        )
        self._queue_thumb_thread.start()

    def _queue_thumb_worker(self, photos: list, generation: int) -> None:
        """Downscale *photos* to queue thumbnails on a background thread."""
        textures: list[Gdk.Texture | None] = []
        for photo in photos:
            try:
                textures.append(_load_thumbnail(photo.path))
            except Exception:  # a bad image must never crash the queue
                log.warning("Could not load queue thumbnail for %s", photo.path)
                textures.append(None)
        GLib.idle_add(self._apply_queue_thumbnails, textures, generation)

    def _apply_queue_thumbnails(
        self, textures: list[Gdk.Texture | None], generation: int
    ) -> bool:
        """Apply downscaled previews to the queue cards (main thread).

        Returns ``False`` so the idle callback runs only once. A stale load
        (the queue was rebuilt since the thumbnails were generated) is
        discarded.
        """
        if generation != self._queue_thumb_generation:
            return False
        for flow_child, texture in zip(self._queue_flow_children(), textures):
            flow_child.picture.set_paintable(texture)
        return False

    def _queue_flow_children(self) -> list[Gtk.FlowBoxChild]:
        """Return the FlowBox children currently in the queue tab."""
        rows: list[Gtk.FlowBoxChild] = []
        child = self.queue_flow.get_first_child()
        while child is not None:
            rows.append(child)
            child = child.get_next_sibling()
        return rows

    def _on_queue_remove_clicked(self, button: Gtk.Button) -> None:
        """Remove the queue item whose card contains *button*.

        Only the staged item is removed; the original archive file is never
        touched.
        """
        card = button.get_parent()
        flow_child = card.get_parent() if card is not None else None
        item = getattr(flow_child, "item", None)
        if item is None:
            return
        self._queue.remove(item)
        self._queue_by_path.pop(str(item.photo.path), None)
        self.queue_flow.remove(flow_child)
        self._log_message(f"Removed {item.photo.name} from the queue.")
        self._update_queue_ui()

    def _on_discard_queue_clicked(self, _button: Gtk.Button | None) -> None:
        """Clear the queue after confirmation (no files are moved)."""
        if not self._queue:
            return
        if not self._confirm_discard_queue():
            return
        self._queue.clear()
        self._queue_by_path.clear()
        self.queue_flow.remove_all()
        self._log_message("Queue discarded.")
        self._update_queue_ui()

    def _on_execute_queue_clicked(self, _button: Gtk.Button | None) -> None:
        """Move every queued file to the dated trash folder after confirmation.

        The move runs on a background thread so the GTK main loop stays
        responsive; the result is marshalled back with ``GLib.idle_add``.
        Successfully moved files are removed from the queue; files that could
        not be moved stay staged so the user can retry.
        """
        if not self._queue:
            return
        if not self._confirm_execute_queue():
            return
        items = list(self._queue)
        trash_root = self.cfg.resolved_trash_root()
        self._log_message(f"Moving {len(items)} file(s) to trash...")
        self._execution_thread = threading.Thread(
            target=self._execution_worker,
            args=(items, trash_root),
            daemon=True,
        )
        self._execution_thread.start()

    def _execution_worker(self, items: list[QueueItem], trash_root: Path) -> None:
        """Move *items* to trash off the main thread and marshal the result back."""
        try:
            result = move_to_trash(items, trash_root)
        except Exception as exc:  # a trash failure must never kill the app
            log.exception("Trash execution failed")
            result = TrashResult(
                failures=[
                    TrashFailure(item=item, error=f"Execution failed: {exc}")
                    for item in items
                ]
            )
        GLib.idle_add(self._finish_execution, result)

    def _finish_execution(self, result: TrashResult) -> bool:
        """Apply an execution result to the queue and view (main thread).

        Successfully moved files are removed from the queue; failed files stay
        staged. The current review grid is refreshed so moved files disappear
        from the active Similarity/Blur view. Returns ``False`` so the idle
        callback runs only once.
        """
        moved_paths = {entry.original_path for entry in result.moved}
        if moved_paths:
            self._queue = [item for item in self._queue if str(item.photo.path) not in moved_paths]
            self._queue_by_path = {str(item.photo.path): item for item in self._queue}
            self.queue_flow.remove_all()
            for item in self._queue:
                self.queue_flow.append(self._build_queue_card(item))
            self._log_message(
                f"Moved {len(result.moved)} file(s) to trash "
                f"({self.cfg.resolved_trash_root() / 'YYYY-MM-DD'}/trash.log.json)."
            )
        for failure in result.failures:
            self._log_message(f"Could not move {failure.item.photo.name}: {failure.error}")
        if result.failures:
            self._log_message(
                f"{len(result.failures)} file(s) could not be moved and remain in the queue."
            )
        self._update_queue_ui()
        if moved_paths:
            self._load_queue_thumbnails()
        self._refresh_current_result()
        return False

    def _confirm_execute_queue(self) -> bool:
        """Ask the user to confirm moving the whole queue to trash.

        Returns True when the user confirms. The dialog shows the number of
        queued files and the total size about to be moved.
        """
        count = len(self._queue)
        total = sum(item.photo.size for item in self._queue)
        dialog = Gtk.AlertDialog(
            message=f"Move {count} file(s) to trash?",
            detail=(
                f"Total size: {_human_size(total)}. "
                "Files will be moved to the trash folder and can be restored from there."
            ),
        )
        dialog.set_buttons(["Cancel", "Move to Trash"])
        dialog.set_cancel_button(0)
        dialog.set_default_button(1)
        return dialog.choose(self, None, None, None) == 1

    def _confirm_discard_queue(self) -> bool:
        """Ask the user to confirm discarding the whole queue.

        Returns True when the user confirms. No files are moved either way.
        """
        dialog = Gtk.AlertDialog(
            message="Discard the queue?",
            detail="The staged files will be removed from the queue. No files are moved or deleted.",
        )
        dialog.set_buttons(["Cancel", "Discard Queue"])
        dialog.set_cancel_button(0)
        dialog.set_default_button(1)
        return dialog.choose(self, None, None, None) == 1

    def _update_queue_ui(self) -> None:
        """Refresh the queue header, empty-state label, and button sensitivity."""
        count = len(self._queue)
        total = sum(item.photo.size for item in self._queue)
        self.queue_count_label.set_text(f"Queue: {count} items ({_human_size(total)})")
        self.queue_empty_label.set_visible(count == 0)
        self.execute_queue_button.set_sensitive(count > 0)
        self.discard_queue_button.set_sensitive(count > 0)

    def _clear_selection(self) -> None:
        """Clear the current grid selection and update the UI."""
        for cell in self._grid_cells:
            if cell.photo is not None:
                cell.checkbox.set_active(False)
        self._selection = set()
        self._update_selection_buttons()
        self._update_status()

    def _refresh_current_result(self) -> None:
        """Reload the current result row so moved files disappear from the grid.

        After an execution, files that were moved to trash no longer exist at
        their original paths. Reloading the current result row rebuilds the
        grid from the live result data; clusters or candidates whose members
        were all moved are removed from the result list.
        """
        if self._current_result_id is None:
            return
        row = self.result_list.get_selected_row()
        if row is None:
            return
        result = getattr(row, "result", None)
        if isinstance(result, Cluster):
            remaining = [m for m in result.members if m.path.exists()]
            if not remaining:
                self.result_list.remove(row)
                self._clear_grid()
                self._log_message("Removed an empty cluster from the result list.")
                return
            result.members = remaining
            row.title = cluster_label(result, self._row_index(row) + 1)
            label = row.get_child()
            if isinstance(label, Gtk.Label):
                label.set_text(row.title)
            self._load_grid(row.result_id, remaining)
        elif isinstance(result, BlurCandidate):
            if not result.photo.path.exists():
                self.result_list.remove(row)
                self._clear_grid()
                self._log_message("Removed an empty candidate from the result list.")
                return
            self._load_grid(row.result_id, [result.photo], tooltips=[f"Blur score {result.score:.1f}, {result.percentile:.0f}%"])

    def _row_index(self, row: Gtk.ListBoxRow) -> int:
        """Return the 0-based index of *row* in the result list."""
        index = 0
        child = self.result_list.get_first_child()
        while child is not None:
            if child is row:
                return index
            index += 1
            child = child.get_next_sibling()
        return 0

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
