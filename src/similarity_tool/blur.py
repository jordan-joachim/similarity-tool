"""Blur / shaky detection using Laplacian-variance sharpness scoring.

Every image is scored with OpenCV's Laplacian variance: the image is decoded
to grayscale, convolved with a 3x3 Laplacian kernel, and the variance of the
result is the sharpness score (higher = sharper). Scores are cached in the
same SQLite database as perceptual hashes, keyed by absolute path and file
mtime, so unchanged files reuse their cached score. Scoring runs across all
available CPU cores via a ``ProcessPoolExecutor``.

Candidate selection returns the bottom ``blur_threshold_percentile`` percent
of a month's images by score plus any image whose score is below
``blur_min_absolute``. The thresholds are read from the config at scan time,
so editing ``config.json`` changes the candidate set of the next scan.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from similarity_tool.config import Config
from similarity_tool.hashing import HashCache, default_worker_count, update_blur_scores
from similarity_tool.models import BlurCandidate, HashRecord, PhotoFile

log = logging.getLogger(__name__)

#: Message shown when Blur mode is disabled in the configuration.
BLUR_DISABLED_MESSAGE = "Blur mode is disabled in the configuration"

#: Message shown when a scan finds no blurry candidates.
EMPTY_STATE_MESSAGE = "No blurry images found"


def blur_score(path: str | Path) -> float | None:
    """Return the Laplacian-variance sharpness score for *path*.

    Higher scores mean sharper images. Returns ``None`` (and logs a warning)
    when the image cannot be decoded.
    """
    import cv2  # imported lazily so the module imports without it

    try:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("could not decode image")
        return float(cv2.Laplacian(image, cv2.CV_64F).var())
    except (OSError, ValueError) as exc:
        log.warning("Skipping unreadable image %s: %s", path, exc)
        return None


def _score_worker(path: str) -> tuple[float | None, str | None]:
    """Module-level wrapper around :func:`blur_score` for the process pool.

    Kept at module level so it can be pickled by ``ProcessPoolExecutor``. The
    error message is returned (not logged) so the parent process can report it
    through its own log handlers, which is what the GUI Log panel sees.
    """
    try:
        score = blur_score(path)
    except Exception as exc:  # noqa: BLE001 - a worker must never kill the pool
        return None, str(exc)
    if score is None:
        return None, "could not decode image"
    return score, None


class BlurScorer:
    """SQLite-backed cache of Laplacian-variance scores keyed by path and mtime.

    Scores share the ``hashes`` table with perceptual hashes (the
    ``blur_score`` column), so a single cache serves both detection modes.
    """

    def __init__(
        self,
        db_path: str | Path,
        max_workers: int | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.max_workers = max_workers if max_workers is not None else default_worker_count()
        # HashCache is thread-safe (check_same_thread=False + internal lock),
        # so the GUI can call compute_scores() from a worker thread.
        self._cache = HashCache(self.db_path)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._cache.close()

    def compute_scores(self, photos: Sequence[PhotoFile]) -> list[HashRecord]:
        """Return blur-score records for *photos*, reusing the cache where possible.

        Files whose cached mtime matches the current mtime are reused without
        recomputation. Files with a changed mtime (or no cache entry) are
        scored in parallel across the configured worker processes. Corrupt
        images are skipped and logged; they produce no record. The returned
        list preserves the input order of *photos* (skipping corrupt files).
        """
        records_by_path: dict[str, HashRecord] = {}
        to_score: list[PhotoFile] = []
        for photo in photos:
            cached = self._cache.get(photo.path, photo.mtime)
            if cached is not None and cached.blur_score is not None:
                records_by_path[cached.photo_path] = cached
            else:
                to_score.append(photo)

        if to_score:
            for record in self._score_parallel(to_score):
                update_blur_scores(self._cache, [record])
                records_by_path[record.photo_path] = record

        return [
            records_by_path[str(photo.path)]
            for photo in photos
            if str(photo.path) in records_by_path
        ]

    def _score_parallel(self, photos: Sequence[PhotoFile]) -> list[HashRecord]:
        """Compute blur scores for *photos* using a process pool."""
        results: list[tuple[float | None, str | None]] = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(_score_worker, [str(p.path) for p in photos]))

        records: list[HashRecord] = []
        for photo, (score, error) in zip(photos, results):
            if score is None:
                log.warning("Skipping unreadable image %s: %s", photo.path, error)
                continue
            records.append(
                HashRecord(
                    photo_path=str(photo.path),
                    mtime=photo.mtime,
                    blur_score=score,
                )
            )
        return records


def select_candidates(
    records: Sequence[HashRecord],
    photos: Sequence[PhotoFile],
    percentile: float,
    min_absolute: float,
) -> list[BlurCandidate]:
    """Select blurry candidates from scored records.

    Returns the bottom *percentile* percent of images by score (at least one
    image when the percentile is positive and images exist) plus every image
    whose score is below *min_absolute*. Candidates are ordered by increasing
    score (most blurry first) and carry their percentile rank. Records without
    a score, and records without a matching photo, are ignored.
    """
    photo_by_path = {str(photo.path): photo for photo in photos}
    scored: list[tuple[PhotoFile, float]] = []
    for record in records:
        if record.blur_score is None:
            continue
        photo = photo_by_path.get(record.photo_path)
        if photo is None:
            continue
        scored.append((photo, record.blur_score))

    if not scored:
        return []

    scored.sort(key=lambda item: item[1])
    n = len(scored)
    selected: set[int] = set()

    if percentile > 0:
        count = max(1, math.ceil(n * percentile / 100.0))
        selected.update(range(min(count, n)))

    for index, (_, score) in enumerate(scored):
        if score < min_absolute:
            selected.add(index)

    candidates: list[BlurCandidate] = []
    for index in sorted(selected):
        photo, score = scored[index]
        candidates.append(
            BlurCandidate(
                photo=photo,
                score=score,
                percentile=round((index + 1) / n * 100.0, 1),
            )
        )
    return candidates


@dataclass
class BlurResult:
    """The outcome of a blur scan.

    ``candidates`` holds the blurry images ordered most-blurry-first.
    ``empty_message`` is the empty-state text the GUI should show when no
    candidates were found (or Blur mode is disabled).
    """

    candidates: list[BlurCandidate] = field(default_factory=list)
    empty_message: str = EMPTY_STATE_MESSAGE

    @property
    def is_empty(self) -> bool:
        return not self.candidates


def scan_blur(
    photos: Sequence[PhotoFile],
    cfg: Config,
    db_path: str | Path | None = None,
    max_workers: int | None = None,
) -> BlurResult:
    """Score *photos* and return the blurry candidates for the configured month.

    The thresholds are read from *cfg* at call time, so editing
    ``config.json`` and restarting the application changes the candidate set of
    the next scan. When ``blur_enabled`` is false, no scoring happens and an
    empty result with a disabled-state message is returned.
    """
    if not cfg.blur_enabled:
        return BlurResult(empty_message=BLUR_DISABLED_MESSAGE)

    db = Path(db_path) if db_path is not None else cfg.resolved_cache_path()
    scorer = BlurScorer(db, max_workers=max_workers)
    try:
        records = scorer.compute_scores(photos)
    finally:
        scorer.close()

    candidates = select_candidates(
        records,
        photos,
        percentile=cfg.blur_threshold_percentile,
        min_absolute=cfg.blur_min_absolute,
    )
    return BlurResult(candidates=candidates)
