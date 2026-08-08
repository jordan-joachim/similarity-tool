"""Tests for clustering similar images from perceptual-hash distances.

The clustering module groups images whose pHash distance is within
``phash_threshold`` AND whose dHash distance is within ``dhash_threshold``.
These tests use synthetic hash records with hand-crafted Hamming distances so
the threshold behavior is exact and deterministic, plus one end-to-end test
with real images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from similarity_tool.clusters import (
    EMPTY_STATE_MESSAGE,
    MAX_CLUSTER_SIZE,
    build_clusters,
    within_thresholds,
)
from similarity_tool.hashing import hash_image
from similarity_tool.models import HashRecord, PhotoFile


def _hex(value: int) -> str:
    """Format an integer as a 16-character (64-bit) hex hash."""
    return f"{value:016x}"


def _record(path: str, phash: int, dhash: int, mtime: float = 1.0) -> HashRecord:
    return HashRecord(photo_path=path, mtime=mtime, phash=_hex(phash), dhash=_hex(dhash))


def _photo(path: str, size: int = 100) -> PhotoFile:
    return PhotoFile(path=Path(path), relative_path=path, size=size, mtime=1.0)


def _items(paths, phashes, dhashes):
    """Build parallel (records, photos) lists from hash integers."""
    photos = [_photo(p) for p in paths]
    records = [_record(p, ph, dh) for p, ph, dh in zip(paths, phashes, dhashes)]
    return records, photos


class TestWithinThresholds:
    def test_matrix_shape_and_false_diagonal(self):
        records = [_record("/a", 0, 0), _record("/b", 1, 1), _record("/c", 0x10, 0x10)]
        within = within_thresholds(records, 8, 10)
        assert within.shape == (3, 3)
        assert within.dtype == np.bool_
        assert not np.any(np.diag(within))

    def test_pair_within_both_thresholds(self):
        records = [_record("/a", 0, 0), _record("/b", 1, 1)]
        within = within_thresholds(records, 8, 10)
        assert within[0, 1] and within[1, 0]

    def test_pair_outside_phash_threshold(self):
        # 0x1FF has 9 bits set: within dhash (9 <= 10) but outside phash (9 > 8).
        records = [_record("/a", 0, 0), _record("/b", 0x1FF, 1)]
        within = within_thresholds(records, 8, 10)
        assert not within[0, 1]

    def test_pair_outside_dhash_threshold(self):
        # 0x7FF has 11 bits set: within phash (1 <= 8) but outside dhash (11 > 10).
        records = [_record("/a", 0, 0), _record("/b", 1, 0x7FF)]
        within = within_thresholds(records, 8, 10)
        assert not within[0, 1]

    def test_empty_hash_never_matches(self):
        records = [
            HashRecord(photo_path="/a", mtime=1.0, phash="", dhash=""),
            _record("/b", 0, 0),
        ]
        within = within_thresholds(records, 8, 10)
        assert not within[0, 1]

    def test_malformed_hash_never_matches(self):
        records = [
            HashRecord(photo_path="/a", mtime=1.0, phash="zz" * 8, dhash="0" * 16),
            _record("/b", 0, 0),
        ]
        within = within_thresholds(records, 8, 10)
        assert not within[0, 1]

    def test_single_record_returns_false_matrix(self):
        within = within_thresholds([_record("/a", 0, 0)], 8, 10)
        assert within.shape == (1, 1)
        assert not within[0, 0]


class TestBuildClusters:
    def test_empty_records_returns_empty_result(self):
        result = build_clusters([], [], 8, 10)
        assert result.clusters == []
        assert result.empty_message == EMPTY_STATE_MESSAGE
        assert result.is_empty

    def test_single_photo_produces_no_cluster(self):
        records, photos = _items(["/a"], [0], [0])
        result = build_clusters(records, photos, 8, 10)
        assert result.clusters == []
        assert result.empty_message == EMPTY_STATE_MESSAGE

    def test_two_similar_photos_form_one_cluster(self):
        records, photos = _items(["/a", "/b"], [0, 1], [0, 1])
        result = build_clusters(records, photos, 8, 10)
        assert len(result.clusters) == 1
        assert {m.path for m in result.clusters[0].members} == {Path("/a"), Path("/b")}

    def test_phash_match_alone_is_not_enough(self):
        # phash distance 1 (within 8) but dhash distance 11 (outside 10).
        records, photos = _items(["/a", "/b"], [0, 1], [0, 0x7FF])
        result = build_clusters(records, photos, 8, 10)
        assert result.clusters == []

    def test_dhash_match_alone_is_not_enough(self):
        # dhash distance 1 (within 10) but phash distance 9 (outside 8).
        records, photos = _items(["/a", "/b"], [0, 0x1FF], [0, 1])
        result = build_clusters(records, photos, 8, 10)
        assert result.clusters == []

    def test_unrelated_photos_produce_empty_state(self):
        # Three hashes pairwise more than 8 bits apart in both algorithms.
        records, photos = _items(
            ["/a", "/b", "/c"],
            [0, 0xFFFF, 0xFFFF0000],
            [0, 0xFFFF, 0xFFFF0000],
        )
        result = build_clusters(records, photos, 8, 10)
        assert result.clusters == []
        assert result.empty_message == EMPTY_STATE_MESSAGE

    def test_cluster_members_are_pairwise_within_thresholds(self):
        # Four images whose hashes differ by at most 3 bits in each algorithm.
        records, photos = _items(
            ["/a", "/b", "/c", "/d"],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        )
        result = build_clusters(records, photos, 8, 10)
        assert len(result.clusters) == 1
        members = result.clusters[0].members
        assert len(members) == 4
        # Every pair in the cluster is within both thresholds.
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                p = int(records[i].phash, 16) ^ int(records[j].phash, 16)
                d = int(records[i].dhash, 16) ^ int(records[j].dhash, 16)
                assert p.bit_count() <= 8
                assert d.bit_count() <= 10

    def test_cluster_never_contains_violating_pair(self):
        # A 4-cycle: A-B, B-C, C-D, A-D are within thresholds; A-C and B-D are
        # not. The splitter must never put a violating pair in one cluster.
        records, photos = _items(
            ["/a", "/b", "/c", "/d"],
            [0x0000, 0x00FF, 0xFFFF, 0xFF00],
            [0x0000, 0x00FF, 0xFFFF, 0xFF00],
        )
        result = build_clusters(records, photos, 8, 10)
        covered = {m.path for c in result.clusters for m in c.members}
        assert covered == {Path("/a"), Path("/b"), Path("/c"), Path("/d")}
        for cluster in result.clusters:
            paths = [m.path for m in cluster.members]
            if Path("/a") in paths and Path("/c") in paths:
                pytest.fail("cluster contains A and C, which violate the thresholds")
            if Path("/b") in paths and Path("/d") in paths:
                pytest.fail("cluster contains B and D, which violate the thresholds")

    def test_burst_of_12_is_split_into_clusters_of_at_most_8(self):
        paths = [f"/img{i}" for i in range(12)]
        records, photos = _items(paths, [0] * 12, [0] * 12)
        result = build_clusters(records, photos, 8, 10)
        assert len(result.clusters) >= 2
        assert all(len(c.members) <= MAX_CLUSTER_SIZE for c in result.clusters)
        assert all(len(c.members) >= 2 for c in result.clusters)
        assert sum(len(c.members) for c in result.clusters) == 12

    def test_custom_max_size(self):
        paths = [f"/img{i}" for i in range(10)]
        records, photos = _items(paths, [0] * 10, [0] * 10)
        result = build_clusters(records, photos, 8, 10, max_size=4)
        assert all(len(c.members) <= 4 for c in result.clusters)
        assert sum(len(c.members) for c in result.clusters) == 10

    def test_photo_without_record_is_ignored(self):
        records, photos = _items(["/a", "/b"], [0, 1], [0, 1])
        photos.append(_photo("/corrupt"))  # no hash record, e.g. corrupt image
        result = build_clusters(records, photos, 8, 10)
        assert len(result.clusters) == 1
        assert all(m.path != Path("/corrupt") for m in result.clusters[0].members)

    def test_record_without_photo_is_ignored(self):
        records, photos = _items(["/a"], [0], [0])
        records.append(_record("/ghost", 0, 0))
        result = build_clusters(records, photos, 8, 10)
        assert result.clusters == []

    def test_results_are_deterministic(self):
        records, photos = _items(
            ["/a", "/b", "/c", "/d"],
            [0x0000, 0x00FF, 0xFFFF, 0xFF00],
            [0x0000, 0x00FF, 0xFFFF, 0xFF00],
        )
        first = build_clusters(records, photos, 8, 10)
        second = build_clusters(records, photos, 8, 10)
        assert [[m.path for m in c.members] for c in first.clusters] == [
            [m.path for m in c.members] for c in second.clusters
        ]

    def test_cluster_carries_hash_algorithms(self):
        records, photos = _items(["/a", "/b"], [0, 1], [0, 1])
        result = build_clusters(records, photos, 8, 10, hash_algorithms=["phash", "dhash"])
        assert result.clusters[0].hash_algorithms == ["phash", "dhash"]

    def test_largest_clusters_sort_first(self):
        # A burst of 10 identical images plus a pair of near-duplicates.
        paths = [f"/burst{i}" for i in range(10)] + ["/pair_a", "/pair_b"]
        phashes = [0] * 10 + [0x100, 0x101]
        dhashes = [0] * 10 + [0x100, 0x101]
        records, photos = _items(paths, phashes, dhashes)
        result = build_clusters(records, photos, 8, 10)
        sizes = [len(c.members) for c in result.clusters]
        assert sizes == sorted(sizes, reverse=True)

    def test_large_month_stays_bounded(self):
        # 3000 images with random hashes: clustering must complete quickly and
        # the pairwise matrix must be a compact boolean array, not a Python
        # object per pair.
        rng = np.random.default_rng(7)
        n = 3000
        paths = [f"/img{i}" for i in range(n)]
        phashes = [int(v) for v in rng.integers(0, 2**64, size=n, dtype=np.uint64)]
        dhashes = [int(v) for v in rng.integers(0, 2**64, size=n, dtype=np.uint64)]
        records, photos = _items(paths, phashes, dhashes)
        result = build_clusters(records, photos, 8, 10)
        # Random 64-bit hashes essentially never cluster at these thresholds.
        assert result.clusters == []
        assert result.empty_message == EMPTY_STATE_MESSAGE

    def test_large_month_with_burst(self):
        # 2000 unrelated images plus a burst of 20 identical ones.
        rng = np.random.default_rng(3)
        n = 2000
        paths = [f"/img{i}" for i in range(n)]
        phashes = [int(v) for v in rng.integers(0, 2**64, size=n, dtype=np.uint64)]
        dhashes = [int(v) for v in rng.integers(0, 2**64, size=n, dtype=np.uint64)]
        for i in range(20):
            paths.append(f"/burst{i}")
            phashes.append(0)
            dhashes.append(0)
        records, photos = _items(paths, phashes, dhashes)
        result = build_clusters(records, photos, 8, 10)
        burst_clusters = [
            c for c in result.clusters if c.members[0].path.name.startswith("burst")
        ]
        assert sum(len(c.members) for c in burst_clusters) == 20
        assert all(len(c.members) <= MAX_CLUSTER_SIZE for c in result.clusters)


class TestEndToEnd:
    def test_identical_images_cluster_and_different_image_does_not(self, tmp_path):
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        other = np.random.default_rng(99).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        a = tmp_path / "a.jpg"
        b = tmp_path / "b.jpg"
        c = tmp_path / "c.jpg"
        Image.fromarray(arr).save(a, format="JPEG")
        Image.fromarray(arr).save(b, format="JPEG")  # identical bytes to a
        Image.fromarray(other).save(c, format="JPEG")  # unrelated image

        photos = []
        for p in (a, b, c):
            stat = p.stat()
            photos.append(
                PhotoFile(
                    path=p,
                    relative_path=p.as_posix(),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                )
            )

        records = []
        for photo in photos:
            hashes = hash_image(photo.path, ["phash", "dhash"])
            records.append(
                HashRecord(
                    photo_path=str(photo.path),
                    mtime=photo.mtime,
                    phash=hashes["phash"],
                    dhash=hashes["dhash"],
                )
            )

        result = build_clusters(records, photos, 8, 10)
        assert len(result.clusters) == 1
        members = {m.path for m in result.clusters[0].members}
        assert Path(a) in members and Path(b) in members
        assert Path(c) not in members
