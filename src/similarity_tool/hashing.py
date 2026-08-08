"""Perceptual hashing with an SQLite cache and parallel computation.

This module computes pHash and dHash values for images using ``imagehash``
and stores them in an SQLite database keyed by absolute path and file mtime.
Unchanged files reuse their cached hashes; files whose mtime changed are
rehashed. Hashing runs across all available CPU cores via a
``ProcessPoolExecutor``. Images that cannot be decoded are skipped and logged
rather than aborting the scan.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from similarity_tool.models import HashRecord, PhotoFile

log = logging.getLogger(__name__)

# imagehash algorithms this module knows how to compute and store.
_ALGORITHMS = ("phash", "dhash")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hashes (
    path        TEXT PRIMARY KEY,
    mtime       REAL NOT NULL,
    phash       TEXT NOT NULL DEFAULT '',
    dhash       TEXT NOT NULL DEFAULT '',
    blur_score  REAL,
    ai_embedding BLOB
)
"""


def default_worker_count() -> int:
    """Return the number of worker processes to use for hashing.

    Uses all available CPU cores, falling back to 1 if the OS reports none.
    """
    return os.cpu_count() or 1


def hamming_distance(left: str, right: str) -> int:
    """Return the Hamming distance between two hex-encoded hashes.

    Both hashes must be the same length and non-empty; a ``ValueError`` is
    raised otherwise.
    """
    if not left or not right:
        raise ValueError("hashes must not be empty")
    if len(left) != len(right):
        raise ValueError("hashes must have the same length")
    return sum((int(a, 16) ^ int(b, 16)).bit_count() for a, b in zip(left, right))


def _compute_hashes(
    path: str | Path, algorithms: Sequence[str]
) -> tuple[dict[str, str] | None, str | None]:
    """Compute the requested perceptual hashes for *path*.

    Returns ``(hashes, error)`` where *hashes* is a mapping of algorithm name
    to hex-encoded hash, or ``None`` with a human-readable *error* if the image
    cannot be decoded. No logging happens here so callers can decide where the
    failure is reported (the process pool parent logs it, not the worker).
    """
    import imagehash  # imported lazily so the module imports without it

    try:
        with Image.open(path) as image:
            image.load()
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
        return None, str(exc)

    hashes: dict[str, str] = {}
    for algorithm in algorithms:
        if algorithm == "phash":
            hashes[algorithm] = str(imagehash.phash(image))
        elif algorithm == "dhash":
            hashes[algorithm] = str(imagehash.dhash(image))
        else:
            raise ValueError(f"unsupported hash algorithm: {algorithm}")
    return hashes, None


def hash_image(path: str | Path, algorithms: Sequence[str]) -> dict[str, str] | None:
    """Compute the requested perceptual hashes for *path*.

    Returns a mapping of algorithm name to hex-encoded hash, or ``None`` if
    the image cannot be decoded (in which case a warning is logged).
    """
    hashes, error = _compute_hashes(path, algorithms)
    if hashes is None:
        log.warning("Skipping unreadable image %s: %s", path, error)
        return None
    return hashes


def _hash_worker(
    path: str, algorithms: tuple[str, ...]
) -> tuple[dict[str, str] | None, str | None]:
    """Module-level wrapper around :func:`_compute_hashes` for the process pool.

    Kept at module level so it can be pickled by ``ProcessPoolExecutor``. The
    error message is returned (not logged) so the parent process can report it
    through its own log handlers, which is what the GUI Log panel sees.
    """
    return _compute_hashes(path, algorithms)


class HashCache:
    """SQLite-backed cache of perceptual hashes keyed by path and mtime.

    The cache lives in a single SQLite database (``hashes.sqlite3`` under
    ``~/.cache/similarity-tool/`` by default). The parent directory is created
    automatically on first use.
    """

    def __init__(
        self,
        db_path: str | Path,
        algorithms: Sequence[str] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.algorithms = tuple(algorithms) if algorithms is not None else _ALGORITHMS
        for algorithm in self.algorithms:
            if algorithm not in _ALGORITHMS:
                raise ValueError(f"unsupported hash algorithm: {algorithm}")
        self.max_workers = max_workers if max_workers is not None else default_worker_count()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False lets the GUI call compute_hashes() from a
        # worker thread; a lock serializes access to the connection.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def get(self, path: str | Path, mtime: float) -> HashRecord | None:
        """Return the cached record for *path* if its mtime matches, else ``None``."""
        with self._lock:
            row = self._conn.execute(
                "SELECT path, mtime, phash, dhash, blur_score, ai_embedding"
                " FROM hashes WHERE path = ?",
                (str(path),),
            ).fetchone()
        if row is None:
            return None
        record = HashRecord(
            photo_path=row[0],
            mtime=row[1],
            phash=row[2] or "",
            dhash=row[3] or "",
            blur_score=row[4],
            ai_embedding=row[5] or b"",
        )
        if record.mtime != mtime:
            return None
        return record

    def put(self, record: HashRecord) -> None:
        """Insert or update *record*, preserving any existing blur score."""
        with self._lock:
            existing = self._conn.execute(
                "SELECT blur_score, ai_embedding FROM hashes WHERE path = ?",
                (record.photo_path,),
            ).fetchone()
            blur_score = record.blur_score
            ai_embedding = record.ai_embedding
            if existing is not None:
                if blur_score is None:
                    blur_score = existing[0]
                if not ai_embedding:
                    ai_embedding = existing[1] or b""
            self._conn.execute(
                "INSERT OR REPLACE INTO hashes"
                " (path, mtime, phash, dhash, blur_score, ai_embedding)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.photo_path,
                    record.mtime,
                    record.phash,
                    record.dhash,
                    blur_score,
                    ai_embedding,
                ),
            )
            self._conn.commit()

    def compute_hashes(self, photos: Sequence[PhotoFile]) -> list[HashRecord]:
        """Return hash records for *photos*, reusing the cache where possible.

        Files whose cached mtime matches the current mtime are reused without
        recomputation. Files with a changed mtime (or no cache entry) are
        hashed in parallel across the configured worker processes. Corrupt
        images are skipped and logged; they produce no record. The returned
        list preserves the input order of *photos* (skipping corrupt files).
        """
        records_by_path: dict[str, HashRecord] = {}
        to_hash: list[PhotoFile] = []
        for photo in photos:
            cached = self.get(photo.path, photo.mtime)
            if cached is not None:
                records_by_path[cached.photo_path] = cached
            else:
                to_hash.append(photo)

        if to_hash:
            for record in self._hash_parallel(to_hash):
                self.put(record)
                records_by_path[record.photo_path] = record

        return [
            records_by_path[str(photo.path)]
            for photo in photos
            if str(photo.path) in records_by_path
        ]

    def _hash_parallel(self, photos: Sequence[PhotoFile]) -> list[HashRecord]:
        """Compute hashes for *photos* using a process pool."""
        results: list[tuple[dict[str, str] | None, str | None]] = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(
                executor.map(_hash_worker, [str(p.path) for p in photos], [self.algorithms] * len(photos))
            )

        records: list[HashRecord] = []
        for photo, (hashes, error) in zip(photos, results):
            if hashes is None:
                log.warning("Skipping unreadable image %s: %s", photo.path, error)
                continue
            records.append(
                HashRecord(
                    photo_path=str(photo.path),
                    mtime=photo.mtime,
                    phash=hashes.get("phash", ""),
                    dhash=hashes.get("dhash", ""),
                )
            )
        return records
