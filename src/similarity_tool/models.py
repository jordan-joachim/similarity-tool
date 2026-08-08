"""Shared data structures for the Similarity Tool.

These dataclasses carry file metadata and detection results between the
scanner, hasher, clusterer, blur scorer, GUI, and trash subsystem. They have no
dependencies on GTK or detection libraries so they can be unit-tested in
isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PhotoFile:
    """A JPEG discovered under ``photo_root/YYYY/MM``."""

    path: Path
    relative_path: str  # path relative to photo_root, e.g. "2024/05/IMG_1.jpg"
    size: int
    mtime: float
    date_taken: str | None = None  # EXIF DateTimeOriginal (or DateTime), ISO format

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class HashRecord:
    """A cached detection result for one photo."""

    photo_path: str
    mtime: float
    phash: str = ""
    dhash: str = ""
    blur_score: float | None = None
    ai_embedding: bytes = b""


@dataclass
class Cluster:
    """A group of visually similar photos."""

    members: list[PhotoFile] = field(default_factory=list)
    hash_algorithms: list[str] = field(default_factory=list)
    ai_score: float | None = None  # present only after AI refinement

    def __len__(self) -> int:
        return len(self.members)


@dataclass
class BlurCandidate:
    """A photo flagged as blurry/shaky."""

    photo: PhotoFile
    score: float
    percentile: float | None = None


@dataclass
class QueueItem:
    """A photo staged for deletion, remembering its source mode."""

    photo: PhotoFile
    mode: str  # "similarity" or "blur"


@dataclass
class DeletionLogEntry:
    """One executed move to the trash folder."""

    original_path: str
    trash_path: str
    mode: str
    timestamp: str  # ISO timestamp of execution
    file_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_path": self.original_path,
            "trash_path": self.trash_path,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "file_size": self.file_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeletionLogEntry":
        return cls(
            original_path=str(data["original_path"]),
            trash_path=str(data["trash_path"]),
            mode=str(data["mode"]),
            timestamp=str(data["timestamp"]),
            file_size=int(data["file_size"]),
        )


@dataclass
class DeletionLog:
    """The JSON log written next to a dated trash folder."""

    entries: list[DeletionLogEntry] = field(default_factory=list)

    def add(self, original_path: Path, trash_path: Path, mode: str,
            timestamp: datetime, file_size: int) -> None:
        self.entries.append(
            DeletionLogEntry(
                original_path=str(original_path),
                trash_path=str(trash_path),
                mode=mode,
                timestamp=timestamp.isoformat(timespec="seconds"),
                file_size=file_size,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeletionLog":
        return cls(
            entries=[DeletionLogEntry.from_dict(e) for e in data.get("entries", [])]
        )
