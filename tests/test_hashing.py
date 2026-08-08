"""Tests for perceptual hashing, the SQLite hash cache, and parallel hashing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from similarity_tool.hashing import (
    HashCache,
    default_worker_count,
    hamming_distance,
    hash_image,
)
from similarity_tool.models import HashRecord, PhotoFile


def _make_image(path: Path, seed: int = 0) -> None:
    """Write a small valid JPEG with deterministic content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


def _photo(path: Path) -> PhotoFile:
    stat = path.stat()
    return PhotoFile(
        path=path,
        relative_path=path.as_posix(),
        size=stat.st_size,
        mtime=stat.st_mtime,
    )


class TestHammingDistance:
    def test_identical_hashes(self):
        assert hamming_distance("0" * 16, "0" * 16) == 0

    def test_single_bit_difference(self):
        assert hamming_distance("0" * 16, "8" + "0" * 15) == 1

    def test_all_bits_differ(self):
        assert hamming_distance("0" * 16, "f" * 16) == 64

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            hamming_distance("abcd", "abc")

    def test_empty_hash_raises(self):
        with pytest.raises(ValueError):
            hamming_distance("", "abcd")


class TestHashImage:
    def test_returns_phash_and_dhash(self, tmp_path):
        img = tmp_path / "a.jpg"
        _make_image(img)
        hashes = hash_image(img, ["phash", "dhash"])
        assert set(hashes) == {"phash", "dhash"}
        assert len(hashes["phash"]) == 16
        assert len(hashes["dhash"]) == 16

    def test_corrupt_image_returns_none_and_logs(self, tmp_path, caplog):
        img = tmp_path / "broken.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0 not a real jpeg")
        with caplog.at_level(logging.WARNING, logger="similarity_tool.hashing"):
            assert hash_image(img, ["phash", "dhash"]) is None
        assert any("Skipping" in record.message for record in caplog.records)

    def test_missing_file_returns_none(self, tmp_path):
        assert hash_image(tmp_path / "nope.jpg", ["phash"]) is None


class TestHashCache:
    def test_creates_cache_directory_automatically(self, tmp_path):
        db = tmp_path / "nested" / "cache" / "hashes.sqlite3"
        cache = HashCache(db)
        try:
            assert db.parent.is_dir()
            assert db.is_file()
        finally:
            cache.close()

    def test_table_schema_has_expected_columns(self, tmp_path):
        cache = HashCache(tmp_path / "hashes.sqlite3")
        try:
            columns = {row[1] for row in cache._conn.execute("PRAGMA table_info(hashes)")}
            assert {"path", "mtime", "phash", "dhash", "blur_score", "ai_embedding"} <= columns
        finally:
            cache.close()

    def test_unknown_algorithm_raises(self, tmp_path):
        with pytest.raises(ValueError):
            HashCache(tmp_path / "h.sqlite3", algorithms=["wavelet"])

    def test_put_and_get_roundtrip(self, tmp_path):
        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            record = HashRecord(
                photo_path="/x/a.jpg", mtime=123.0, phash="a" * 16, dhash="b" * 16
            )
            cache.put(record)
            got = cache.get("/x/a.jpg", 123.0)
            assert got is not None
            assert got.photo_path == "/x/a.jpg"
            assert got.mtime == 123.0
            assert got.phash == "a" * 16
            assert got.dhash == "b" * 16
        finally:
            cache.close()

    def test_get_returns_none_when_mtime_differs(self, tmp_path):
        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            cache.put(
                HashRecord(photo_path="/x/a.jpg", mtime=123.0, phash="a" * 16, dhash="b" * 16)
            )
            assert cache.get("/x/a.jpg", 999.0) is None
        finally:
            cache.close()

    def test_put_preserves_existing_blur_score(self, tmp_path):
        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            cache.put(
                HashRecord(
                    photo_path="/x/a.jpg", mtime=1.0, phash="a" * 16, dhash="b" * 16,
                    blur_score=42.5,
                )
            )
            cache.put(HashRecord(photo_path="/x/a.jpg", mtime=2.0, phash="c" * 16, dhash="d" * 16))
            got = cache.get("/x/a.jpg", 2.0)
            assert got is not None
            assert got.phash == "c" * 16
            assert got.blur_score == 42.5
        finally:
            cache.close()


class TestComputeHashes:
    def test_returns_records_in_input_order(self, tmp_path):
        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            a, b, c = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"
            _make_image(a, seed=1)
            _make_image(b, seed=2)
            _make_image(c, seed=3)
            photos = [_photo(a), _photo(b), _photo(c)]
            records = cache.compute_hashes(photos)
            assert [r.photo_path for r in records] == [str(a), str(b), str(c)]
            assert all(r.phash and r.dhash for r in records)
        finally:
            cache.close()

    def test_cache_hit_reuses_hash_without_recompute(self, tmp_path, monkeypatch):
        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            img = tmp_path / "a.jpg"
            _make_image(img)
            photos = [_photo(img)]
            first = cache.compute_hashes(photos)
            assert len(first) == 1

            def _boom(*args, **kwargs):
                raise AssertionError("recompute should not happen on cache hit")

            monkeypatch.setattr(cache, "_hash_parallel", _boom)
            second = cache.compute_hashes(photos)
            assert second == first
        finally:
            cache.close()

    def test_mtime_change_triggers_rehash(self, tmp_path):
        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            img = tmp_path / "a.jpg"
            _make_image(img)
            first = cache.compute_hashes([_photo(img)])
            os.utime(img, (img.stat().st_atime, img.stat().st_mtime + 1000))
            photos = [_photo(img)]
            second = cache.compute_hashes(photos)
            assert second[0].mtime == photos[0].mtime
            assert second[0].mtime != first[0].mtime
            got = cache.get(img, photos[0].mtime)
            assert got is not None
            assert got.mtime == photos[0].mtime
        finally:
            cache.close()

    def test_only_modified_file_is_rehashed(self, tmp_path, monkeypatch):
        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
            _make_image(a, seed=1)
            _make_image(b, seed=2)
            cache.compute_hashes([_photo(a), _photo(b)])
            os.utime(b, (b.stat().st_atime, b.stat().st_mtime + 1000))
            photos = [_photo(a), _photo(b)]

            hashed: list[Path] = []
            original = cache._hash_parallel

            def recording(plist):
                hashed.extend(p.path for p in plist)
                return original(plist)

            monkeypatch.setattr(cache, "_hash_parallel", recording)
            records = cache.compute_hashes(photos)
            assert hashed == [b]
            assert [r.photo_path for r in records] == [str(a), str(b)]
        finally:
            cache.close()

    def test_mixed_cache_hits_and_misses_preserve_input_order(self, tmp_path, monkeypatch):
        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            a, b, c = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"
            _make_image(a, seed=1)
            _make_image(b, seed=2)
            _make_image(c, seed=3)
            cache.compute_hashes([_photo(a), _photo(b), _photo(c)])
            # Touch only the middle file so it must be rehashed.
            os.utime(b, (b.stat().st_atime, b.stat().st_mtime + 1000))
            photos = [_photo(a), _photo(b), _photo(c)]

            hashed: list[Path] = []
            original = cache._hash_parallel

            def recording(plist):
                hashed.extend(p.path for p in plist)
                return original(plist)

            monkeypatch.setattr(cache, "_hash_parallel", recording)
            records = cache.compute_hashes(photos)
            assert hashed == [b]
            assert [r.photo_path for r in records] == [str(a), str(b), str(c)]
        finally:
            cache.close()

    def test_corrupt_image_is_skipped_and_logged(self, tmp_path, caplog):
        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            good = tmp_path / "good.jpg"
            bad = tmp_path / "bad.jpg"
            _make_image(good)
            bad.write_bytes(b"\xff\xd8\xff\xe0 truncated")
            photos = [_photo(good), _photo(bad)]
            with caplog.at_level(logging.WARNING, logger="similarity_tool.hashing"):
                records = cache.compute_hashes(photos)
            assert [r.photo_path for r in records] == [str(good)]
            assert any("Skipping" in record.message for record in caplog.records)
            assert cache.get(bad, photos[1].mtime) is None
        finally:
            cache.close()

    def test_empty_photo_list_returns_empty(self, tmp_path):
        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            assert cache.compute_hashes([]) == []
        finally:
            cache.close()

    def test_uses_process_pool_with_configured_workers(self, tmp_path, monkeypatch):
        from similarity_tool import hashing

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

        monkeypatch.setattr(hashing, "ProcessPoolExecutor", _RecordingPool)
        cache = HashCache(tmp_path / "h.sqlite3", max_workers=4)
        try:
            imgs = [tmp_path / f"img{i}.jpg" for i in range(3)]
            for i, img in enumerate(imgs):
                _make_image(img, seed=i)
            records = cache.compute_hashes([_photo(i) for i in imgs])
            assert len(records) == 3
            assert _RecordingPool.instances
            assert _RecordingPool.instances[0].max_workers == 4
        finally:
            cache.close()

    def test_parallel_hashing_with_real_process_pool(self, tmp_path):
        cache = HashCache(tmp_path / "h.sqlite3", max_workers=2)
        try:
            imgs = [tmp_path / f"img{i}.jpg" for i in range(4)]
            for i, img in enumerate(imgs):
                _make_image(img, seed=i)
            records = cache.compute_hashes([_photo(i) for i in imgs])
            assert len(records) == 4
            assert all(r.phash and r.dhash for r in records)
        finally:
            cache.close()

    def test_default_worker_count_matches_cpu(self):
        assert default_worker_count() == (os.cpu_count() or 1)

    def test_usable_from_worker_thread(self, tmp_path):
        """The GUI runs scans off the main thread; the cache must not be
        bound to the thread that created it."""
        import threading

        cache = HashCache(tmp_path / "h.sqlite3")
        try:
            img = tmp_path / "a.jpg"
            _make_image(img)
            photo = _photo(img)
            result: list[HashRecord] = []
            error: list[BaseException] = []

            def run():
                try:
                    result.extend(cache.compute_hashes([photo]))
                except BaseException as exc:  # noqa: BLE001
                    error.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            thread.join(timeout=60)
            assert not error, f"worker thread raised: {error}"
            assert len(result) == 1
            assert result[0].photo_path == str(img)
        finally:
            cache.close()
