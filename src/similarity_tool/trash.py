"""Trash subsystem: move files to a dated trash folder and log the moves.

Executed deletions are always moves, never permanent deletes. Each execution
creates (or reuses) a dated folder ``trash_root/YYYY-MM-DD/``, moves every
file into a fresh UUID subfolder preserving its relative path under
``photo_root``, and writes (or appends to) a JSON log next to the dated
folder. Files that fail to move are reported so the caller can keep them in
the queue; the original archive files are never deleted, only moved.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from similarity_tool.models import DeletionLog, DeletionLogEntry, QueueItem

log = logging.getLogger(__name__)

#: Name of the JSON log written next to a dated trash folder.
LOG_FILENAME = "trash.log.json"


@dataclass
class TrashFailure:
    """One file that could not be moved to trash."""

    item: QueueItem
    error: str


@dataclass
class TrashResult:
    """The outcome of one execution batch.

    ``moved`` holds a log entry for every file that was successfully moved;
    ``failures`` lists the files that could not be moved (the caller keeps
    those in the queue).
    """

    moved: list[DeletionLogEntry] = field(default_factory=list)
    failures: list[TrashFailure] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.moved and not self.failures


def dated_trash_dir(trash_root: Path, timestamp: datetime) -> Path:
    """Return the dated trash folder for *timestamp* (``YYYY-MM-DD``)."""
    return trash_root / timestamp.strftime("%Y-%m-%d")


def log_path_for(dated_dir: Path) -> Path:
    """Return the JSON log path next to a dated trash folder."""
    return dated_dir / LOG_FILENAME


def load_log(dated_dir: Path) -> DeletionLog:
    """Load the existing JSON log for *dated_dir*, or an empty log.

    A missing, malformed, or unreadable log yields an empty log so a new
    execution never loses the ability to record its moves.
    """
    path = log_path_for(dated_dir)
    if not path.exists():
        return DeletionLog()
    try:
        with path.open("r", encoding="utf-8") as handle:
            return DeletionLog.from_dict(json.load(handle))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        log.error("Could not read trash log %s (%s); starting a new log.", path, exc)
        return DeletionLog()


def write_log(dated_dir: Path, log_: DeletionLog) -> Path:
    """Write *log_* to the JSON log next to *dated_dir*."""
    path = log_path_for(dated_dir)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(log_.to_dict(), handle, indent=2)
        handle.write("\n")
    return path


def move_to_trash(
    items: list[QueueItem],
    trash_root: Path,
    timestamp: datetime | None = None,
) -> TrashResult:
    """Move *items* to a dated trash folder under *trash_root*.

    Each file is moved into ``trash_root/YYYY-MM-DD/<uuid>/<relative-path>``
    where ``<relative-path>`` mirrors the file's position under ``photo_root``.
    A fresh UUID subfolder is created per execution so same-day batches never
    collide. The JSON log next to the dated folder is appended to (existing
    same-day entries are preserved). Files that fail to move are reported in
    ``TrashResult.failures`` and are not logged; the caller keeps them in the
    queue. The original archive files are never deleted, only moved.
    """
    timestamp = timestamp or datetime.now().astimezone()
    dated_dir = dated_trash_dir(trash_root, timestamp)
    dated_dir.mkdir(parents=True, exist_ok=True)

    batch_uuid = uuid.uuid4().hex
    result = TrashResult()
    for item in items:
        source = item.photo.path
        destination = dated_dir / batch_uuid / item.photo.relative_path
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        except (OSError, shutil.Error) as exc:
            log.error("Could not move %s to trash: %s", source, exc)
            result.failures.append(TrashFailure(item=item, error=str(exc)))
            continue
        result.moved.append(
            DeletionLogEntry(
                original_path=str(source),
                trash_path=str(destination),
                mode=item.mode,
                timestamp=timestamp.isoformat(timespec="seconds"),
                file_size=item.photo.size,
            )
        )

    if result.moved:
        log_ = load_log(dated_dir)
        log_.entries.extend(result.moved)
        write_log(dated_dir, log_)
    return result
