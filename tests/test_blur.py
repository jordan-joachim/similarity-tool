"""Tests for blur/sharpness scoring, the SQLite score cache, and candidate selection."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from PIL import Image

from similarity_tool.blur import (
    BLUR_DISABLED_MESSAGE,
    BlurScorer,
    blur_score,
    scan_blur,
    select_candidates,
)
from similarity_tool.config import Config
from similarity_tool.models import HashRecord, PhotoFile


def _make_image(path: Path, sigma: float = 0.0, seed: int = 0, size: int = 64) -> None:
    """Write a JPEG whose sharpness decreases with *sigma* (Gaussian blur)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    if sigma > 0:
        arr = cv2.GaussianBlur(arr, (0, 0), sigma)
    Image.fromarray(arr).save(path, format="JPEG")


def _photo(path: Path) -> PhotoFile:
    stat = path.stat()
    return PhotoFile(
        path=path,
        relative_path=path.as_posix(),
        size=stat.st_size,
        mtime=stat.st_mtime,
    )


def _record(path: Path, score: float, mtime: float = 1.0) -> HashRecord:
    return HashRecord(photo_path=str(path), mtime=mtime, blur_score=score)


class TestBlurScore:
    def test_sharp_image_scores_higher_than_blurry(self, tmp_path):
        sharp = tmp_path / "sharp.jpg"
        blurry = tmp_path / "blurry.jpg"
        _make_image(sharp, sigma=0.0)
        _make_image(blurry, sigma=6.0)
        sharp_score = blur_score(sharp)
        blurry_score = blur_score(blurry)
        assert sharp_score is not None
        assert blurry_score is not None
        assert sharp_score > blurry_score

    def test_corrupt_image_returns_none_and_logs(self, tmp_path, caplog):
        img = tmp_path / "broken.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0 not a real jpeg")
        with caplog.at_level(logging.WARNING, logger="similarity_tool.blur"):
            assert blur_score(img) is None
        assert any("Skipping" in record.message for record in caplog.records)

    def test_missing_file_returns_none(self, tmp_path):
        assert blur_score(tmp_path / "nope.jpg") is None


class TestBlurScorer:
    def test_scores_every_image_in_input_order(self, tmp_path):
        scorer = BlurScorer(tmp_path / "h.sqlite3")
        try:
            a, b, c = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"
            _make_image(a, sigma=0.0)
            _make_image(b, sigma=2.0)
            _make_image(c, sigma=4.0)
            photos = [_photo(a), _photo(b), _photo(c)]
            records = scorer.compute_scores(photos)
            assert [r.photo_path for r in records] == [str(a), str(b), str(c)]
            assert all(r.blur_score is not None for r in records)
        finally:
            scorer.close()

    def test_every_enumerated_image_has_cached_score(self, tmp_path):
        """VAL-BLUR-006: after a scan every enumerated image has a non-null
        blur score recorded in the cache."""
        scorer = BlurScorer(tmp_path / "h.sqlite3")
        try:
            imgs = [tmp_path / f"img{i}.jpg" for i in range(3)]
            for i, img in enumerate(imgs):
                _make_image(img, sigma=float(i))
            photos = [_photo(i) for i in imgs]
            scorer.compute_scores(photos)
            for photo in photos:
                got = scorer._cache.get(photo.path, photo.mtime)
                assert got is not None
                assert got.blur_score is not None
        finally:
            scorer.close()

    def test_cache_hit_reuses_score_without_recompute(self, tmp_path, monkeypatch):
        scorer = BlurScorer(tmp_path / "h.sqlite3")
        try:
            img = tmp_path / "a.jpg"
            _make_image(img)
            photos = [_photo(img)]
            first = scorer.compute_scores(photos)
            assert len(first) == 1

            def _boom(*args, **kwargs):
                raise AssertionError("recompute should not happen on cache hit")

            monkeypatch.setattr(scorer, "_score_parallel", _boom)
            second = scorer.compute_scores(photos)
            assert second == first
        finally:
            scorer.close()

    def test_mtime_change_triggers_rescore(self, tmp_path):
        scorer = BlurScorer(tmp_path / "h.sqlite3")
        try:
            img = tmp_path / "a.jpg"
            _make_image(img)
            first = scorer.compute_scores([_photo(img)])
            os.utime(img, (img.stat().st_atime, img.stat().st_mtime + 1000))
            photos = [_photo(img)]
            second = scorer.compute_scores(photos)
            assert second[0].mtime == photos[0].mtime
            assert second[0].mtime != first[0].mtime
            got = scorer._cache.get(img, photos[0].mtime)
            assert got is not None
            assert got.blur_score is not None
        finally:
            scorer.close()

    def test_only_modified_file_is_rescored(self, tmp_path, monkeypatch):
        scorer = BlurScorer(tmp_path / "h.sqlite3")
        try:
            a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
            _make_image(a, sigma=0.0)
            _make_image(b, sigma=1.0)
            scorer.compute_scores([_photo(a), _photo(b)])
            os.utime(b, (b.stat().st_atime, b.stat().st_mtime + 1000))
            photos = [_photo(a), _photo(b)]

            scored: list[Path] = []
            original = scorer._score_parallel

            def recording(plist):
                scored.extend(p.path for p in plist)
                return original(plist)

            monkeypatch.setattr(scorer, "_score_parallel", recording)
            records = scorer.compute_scores(photos)
            assert scored == [b]
            assert [r.photo_path for r in records] == [str(a), str(b)]
        finally:
            scorer.close()

    def test_corrupt_image_is_skipped_and_logged(self, tmp_path, caplog):
        scorer = BlurScorer(tmp_path / "h.sqlite3")
        try:
            good = tmp_path / "good.jpg"
            bad = tmp_path / "bad.jpg"
            _make_image(good)
            bad.write_bytes(b"\xff\xd8\xff\xe0 truncated")
            photos = [_photo(good), _photo(bad)]
            with caplog.at_level(logging.WARNING, logger="similarity_tool.blur"):
                records = scorer.compute_scores(photos)
            assert [r.photo_path for r in records] == [str(good)]
            assert any("Skipping" in record.message for record in caplog.records)
            assert scorer._cache.get(bad, photos[1].mtime) is None
        finally:
            scorer.close()

    def test_scoring_preserves_existing_hashes(self, tmp_path):
        scorer = BlurScorer(tmp_path / "h.sqlite3")
        try:
            img = tmp_path / "a.jpg"
            _make_image(img)
            photo = _photo(img)
            scorer._cache.put(
                HashRecord(
                    photo_path=str(img), mtime=photo.mtime, phash="a" * 16, dhash="b" * 16
                )
            )
            records = scorer.compute_scores([photo])
            assert records[0].blur_score is not None
            got = scorer._cache.get(img, photo.mtime)
            assert got is not None
            assert got.phash == "a" * 16
            assert got.dhash == "b" * 16
            assert got.blur_score is not None
        finally:
            scorer.close()

    def test_cache_survives_scorer_restart(self, tmp_path, monkeypatch):
        """VAL-BLUR-019: scores persist across application restarts."""
        db = tmp_path / "h.sqlite3"
        img = tmp_path / "a.jpg"
        _make_image(img)
        photo = _photo(img)

        first = BlurScorer(db)
        try:
            records = first.compute_scores([photo])
            assert len(records) == 1
        finally:
            first.close()

        second = BlurScorer(db)
        try:
            def _boom(*args, **kwargs):
                raise AssertionError("recompute should not happen on cache hit")

            monkeypatch.setattr(second, "_score_parallel", _boom)
            records = second.compute_scores([photo])
            assert len(records) == 1
            assert records[0].blur_score is not None
        finally:
            second.close()

    def test_uses_process_pool_with_configured_workers(self, tmp_path, monkeypatch):
        from similarity_tool import blur

        class _RecordingPool:
            instances: ClassVar[list] = []

            def __init__(self, max_workers=None):
                self.max_workers = max_workers
                _RecordingPool.instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def map(self, fn, *iterables):
                return [fn(*args) for args in zip(*iterables)]

        monkeypatch.setattr(blur, "ProcessPoolExecutor", _RecordingPool)
        scorer = BlurScorer(tmp_path / "h.sqlite3", max_workers=4)
        try:
            imgs = [tmp_path / f"img{i}.jpg" for i in range(3)]
            for i, img in enumerate(imgs):
                _make_image(img, sigma=float(i))
            records = scorer.compute_scores([_photo(i) for i in imgs])
            assert len(records) == 3
            assert _RecordingPool.instances
            assert _RecordingPool.instances[0].max_workers == 4
        finally:
            scorer.close()

    def test_parallel_scoring_with_real_process_pool(self, tmp_path):
        scorer = BlurScorer(tmp_path / "h.sqlite3", max_workers=2)
        try:
            imgs = [tmp_path / f"img{i}.jpg" for i in range(4)]
            for i, img in enumerate(imgs):
                _make_image(img, sigma=float(i))
            records = scorer.compute_scores([_photo(i) for i in imgs])
            assert len(records) == 4
            assert all(r.blur_score is not None for r in records)
        finally:
            scorer.close()

    def test_usable_from_worker_thread(self, tmp_path):
        """The GUI runs scans off the main thread; the scorer must not be
        bound to the thread that created it."""
        import threading

        scorer = BlurScorer(tmp_path / "h.sqlite3")
        try:
            img = tmp_path / "a.jpg"
            _make_image(img)
            photo = _photo(img)
            result: list[HashRecord] = []
            error: list[BaseException] = []

            def run():
                try:
                    result.extend(scorer.compute_scores([photo]))
                except BaseException as exc:  # noqa: BLE001
                    error.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            thread.join(timeout=60)
            assert not error, f"worker thread raised: {error}"
            assert len(result) == 1
            assert result[0].blur_score is not None
        finally:
            scorer.close()


class TestSelectCandidates:
    def _photos(self, tmp_path, names: list[str]) -> list[PhotoFile]:
        photos = []
        for name in names:
            p = tmp_path / name
            p.write_bytes(b"x")
            photos.append(_photo(p))
        return photos

    def test_bottom_percentile_selected(self, tmp_path):
        photos = self._photos(tmp_path, [f"img{i}.jpg" for i in range(10)])
        records = [_record(p.path, float(i)) for i, p in enumerate(photos)]
        candidates = select_candidates(records, photos, percentile=20.0, min_absolute=0.0)
        # Bottom 20% of 10 = 2 images, the two lowest scores.
        assert [c.photo.path for c in candidates] == [photos[0].path, photos[1].path]

    def test_absolute_threshold_includes_low_scores_outside_percentile(self, tmp_path):
        photos = self._photos(tmp_path, [f"img{i}.jpg" for i in range(10)])
        records = [_record(p.path, float(i)) for i, p in enumerate(photos)]
        # Percentile 0 -> no percentile candidates; only scores below 3.0.
        candidates = select_candidates(records, photos, percentile=0.0, min_absolute=3.0)
        assert [c.photo.path for c in candidates] == [photos[0].path, photos[1].path, photos[2].path]

    def test_union_of_percentile_and_absolute(self, tmp_path):
        photos = self._photos(tmp_path, [f"img{i}.jpg" for i in range(10)])
        records = [_record(p.path, float(i)) for i, p in enumerate(photos)]
        # Bottom 10% (1 image) plus anything below 5.0 -> img0..img4.
        candidates = select_candidates(records, photos, percentile=10.0, min_absolute=5.0)
        assert [c.photo.path for c in candidates] == [photos[i].path for i in range(5)]

    def test_sharp_images_outside_thresholds_not_shown(self, tmp_path):
        """VAL-BLUR-008: images above the absolute floor and outside the
        bottom percentile are not candidates."""
        photos = self._photos(tmp_path, [f"img{i}.jpg" for i in range(10)])
        records = [_record(p.path, float(i)) for i, p in enumerate(photos)]
        candidates = select_candidates(records, photos, percentile=10.0, min_absolute=2.0)
        paths = {c.photo.path for c in candidates}
        for i in range(5, 10):
            assert photos[i].path not in paths

    def test_ordered_by_increasing_score(self, tmp_path):
        """VAL-BLUR-014: candidates are ordered most blurry first."""
        photos = self._photos(tmp_path, [f"img{i}.jpg" for i in range(6)])
        records = [_record(p.path, float(i)) for i, p in enumerate(photos)]
        candidates = select_candidates(records, photos, percentile=100.0, min_absolute=0.0)
        scores = [c.score for c in candidates]
        assert scores == sorted(scores)
        assert [c.photo.path for c in candidates] == [photos[i].path for i in range(6)]

    def test_percentile_rank_is_set(self, tmp_path):
        photos = self._photos(tmp_path, [f"img{i}.jpg" for i in range(4)])
        records = [_record(p.path, float(i)) for i, p in enumerate(photos)]
        candidates = select_candidates(records, photos, percentile=100.0, min_absolute=0.0)
        assert [c.percentile for c in candidates] == [25.0, 50.0, 75.0, 100.0]

    def test_empty_inputs_return_empty(self, tmp_path):
        assert select_candidates([], [], percentile=10.0, min_absolute=0.0) == []

    def test_records_without_score_are_ignored(self, tmp_path):
        photos = self._photos(tmp_path, ["a.jpg", "b.jpg"])
        records = [
            _record(photos[0].path, 1.0),
            HashRecord(photo_path=str(photos[1].path), mtime=1.0),
        ]
        candidates = select_candidates(records, photos, percentile=100.0, min_absolute=0.0)
        assert [c.photo.path for c in candidates] == [photos[0].path]

    def test_records_without_matching_photo_are_ignored(self, tmp_path):
        photos = self._photos(tmp_path, ["a.jpg"])
        records = [_record(photos[0].path, 1.0), _record(tmp_path / "ghost.jpg", 0.5)]
        candidates = select_candidates(records, photos, percentile=100.0, min_absolute=0.0)
        assert [c.photo.path for c in candidates] == [photos[0].path]


class TestScanBlur:
    def test_disabled_when_blur_enabled_false(self, tmp_path):
        """VAL-BLUR-002 (module level): a disabled config yields no candidates
        and no scoring."""
        cfg = Config(blur_enabled=False)
        img = tmp_path / "a.jpg"
        _make_image(img)
        result = scan_blur([_photo(img)], cfg, db_path=tmp_path / "h.sqlite3")
        assert result.is_empty
        assert result.empty_message == BLUR_DISABLED_MESSAGE

    def test_uses_config_thresholds(self, tmp_path):
        """VAL-BLUR-020: config changes affect the candidate set of the next scan."""
        db = tmp_path / "h.sqlite3"
        imgs = [tmp_path / f"img{i}.jpg" for i in range(10)]
        for i, img in enumerate(imgs):
            _make_image(img, sigma=float(i))
        photos = [_photo(i) for i in imgs]

        strict = scan_blur(
            photos, Config(blur_threshold_percentile=10.0, blur_min_absolute=0.0), db_path=db
        )
        lenient = scan_blur(
            photos, Config(blur_threshold_percentile=50.0, blur_min_absolute=0.0), db_path=db
        )
        assert len(strict.candidates) == 1  # ceil(10 * 0.10)
        assert len(lenient.candidates) == 5  # ceil(10 * 0.50)
        assert len(strict.candidates) < len(lenient.candidates)

    def test_second_scan_reuses_cache(self, tmp_path, monkeypatch):
        """VAL-BLUR-015: re-scanning unchanged files reuses cached scores."""
        from similarity_tool import blur as blur_mod

        db = tmp_path / "h.sqlite3"
        imgs = [tmp_path / f"img{i}.jpg" for i in range(3)]
        for i, img in enumerate(imgs):
            _make_image(img, sigma=float(i))
        photos = [_photo(i) for i in imgs]
        cfg = Config(blur_threshold_percentile=10.0, blur_min_absolute=0.0)

        first = scan_blur(photos, cfg, db_path=db)
        assert len(first.candidates) == 1

        def _boom(*args, **kwargs):
            raise AssertionError("recompute should not happen on cache hit")

        monkeypatch.setattr(blur_mod.BlurScorer, "_score_parallel", _boom)
        second = scan_blur(photos, cfg, db_path=db)
        assert len(second.candidates) == 1
        assert [c.photo.path for c in second.candidates] == [
            c.photo.path for c in first.candidates
        ]

    def test_empty_photo_list_returns_empty_state(self, tmp_path):
        cfg = Config()
        result = scan_blur([], cfg, db_path=tmp_path / "h.sqlite3")
        assert result.is_empty
        assert result.empty_message == "No blurry images found"
