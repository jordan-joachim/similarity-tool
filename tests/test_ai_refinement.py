"""Tests for the optional CPU-only AI refinement stage.

The AI stage is gated by ``cfg.ai_refinement`` (default off), runs only on
clusters of size 2..=8, uses a CPU-only CLIP model loaded on first use, and
falls back to hash-only clusters when ``transformers``/``torch`` or the model
are unavailable. These tests avoid downloading a real model by injecting a
fake refiner and by monkeypatching the dependency check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from similarity_tool.ai_refinement import (
    DEFAULT_AI_SIMILARITY_THRESHOLD,
    FALLBACK_MESSAGE,
    AIRefiner,
    cluster_label,
    refine_clusters,
    refinement_label,
)
from similarity_tool.clusters import MAX_CLUSTER_SIZE
from similarity_tool.models import Cluster, PhotoFile


def _photo(path: str, size: int = 100) -> PhotoFile:
    return PhotoFile(path=Path(path), relative_path=path, size=size, mtime=1.0)


def _cluster(paths, ai_score=None) -> Cluster:
    return Cluster(
        members=[_photo(p) for p in paths],
        hash_algorithms=["phash", "dhash"],
        ai_score=ai_score,
    )


def _embeddings(vectors) -> np.ndarray:
    """Return unit-norm embedding vectors built from explicit coordinates.

    The real refiner computes cosine similarity as ``embeddings @ embeddings.T``,
    so test fakes must return vectors whose dot products reproduce the intended
    pairwise similarities.
    """
    return np.asarray(vectors, dtype=np.float32)


class _FakeConfig:
    """Minimal stand-in for the Config dataclass."""

    def __init__(self, ai_refinement=False, ai_model="fake/model", ai_similarity_threshold=0.85):
        self.ai_refinement = ai_refinement
        self.ai_model = ai_model
        self.ai_similarity_threshold = ai_similarity_threshold


class _FakeRefiner:
    """Injected refiner that splits clusters based on a similarity matrix."""

    def __init__(self, similarity: np.ndarray, threshold: float = 0.85):
        self.similarity = similarity
        self.threshold = threshold
        self.refined: list[Cluster] = []

    def refine(self, clusters):
        out = []
        for cluster in clusters:
            n = len(cluster.members)
            sub = self.similarity[:n, :n]
            adjacency = sub >= self.threshold
            np.fill_diagonal(adjacency, False)
            # Simple split: keep members pairwise above threshold as one group.
            groups: list[list[int]] = []
            for i in range(n):
                placed = False
                for group in groups:
                    if all(adjacency[i, j] for j in group):
                        group.append(i)
                        placed = True
                        break
                if not placed:
                    groups.append([i])
            for group in groups:
                if len(group) < 2:
                    continue
                members = [cluster.members[i] for i in group]
                out.append(
                    Cluster(
                        members=members,
                        hash_algorithms=list(cluster.hash_algorithms),
                        ai_score=0.9,
                    )
                )
        self.refined = out
        return out


class TestRefineClustersGating:
    def test_disabled_returns_clusters_unchanged(self, monkeypatch):
        """ai_refinement=false must not import transformers/torch or load a model."""
        def _boom(*args, **kwargs):
            raise AssertionError("refiner must not run when AI refinement is disabled")

        monkeypatch.setattr("similarity_tool.ai_refinement.AIRefiner", _boom)
        clusters = [_cluster(["/a", "/b"])]
        result = refine_clusters(clusters, _FakeConfig(ai_refinement=False))
        assert result == clusters

    def test_enabled_runs_refiner(self, monkeypatch):
        """ai_refinement=true invokes the refiner on the hash clusters."""
        fake = _FakeRefiner(np.array([[1.0, 0.9], [0.9, 1.0]]))
        monkeypatch.setattr("similarity_tool.ai_refinement.AIRefiner", lambda *a, **k: fake)
        clusters = [_cluster(["/a", "/b"])]
        result = refine_clusters(clusters, _FakeConfig(ai_refinement=True))
        assert fake.refined == result
        assert len(result) == 1
        assert result[0].is_ai_refined

    def test_empty_cluster_list_returns_empty(self, monkeypatch):
        fake = _FakeRefiner(np.zeros((0, 0)))
        monkeypatch.setattr("similarity_tool.ai_refinement.AIRefiner", lambda *a, **k: fake)
        assert refine_clusters([], _FakeConfig(ai_refinement=True)) == []


class TestAIRefinerFallback:
    def test_missing_dependencies_falls_back(self, monkeypatch, caplog):
        """Missing transformers/torch logs an error and returns hash-only clusters."""
        monkeypatch.setattr(
            "similarity_tool.ai_refinement._dependencies_available", lambda: False
        )
        refiner = AIRefiner("fake/model")
        clusters = [_cluster(["/a", "/b"])]
        with caplog.at_level("ERROR", logger="similarity_tool.ai_refinement"):
            result = refiner.refine(clusters)
        assert result == clusters
        assert any("not installed" in r.message for r in caplog.records)
        assert any(FALLBACK_MESSAGE in r.message for r in caplog.records)

    def test_model_load_failure_falls_back(self, monkeypatch, caplog):
        """A failed model load logs an error and returns hash-only clusters."""
        monkeypatch.setattr(
            "similarity_tool.ai_refinement._dependencies_available", lambda: True
        )

        def _fail_load(self):
            self._load_error = "connection refused"
            log = __import__("logging").getLogger("similarity_tool.ai_refinement")
            log.error("Could not load AI refinement model fake/model: connection refused")

        monkeypatch.setattr(AIRefiner, "_load", _fail_load)
        refiner = AIRefiner("fake/model")
        clusters = [_cluster(["/a", "/b"])]
        with caplog.at_level("ERROR", logger="similarity_tool.ai_refinement"):
            result = refiner.refine(clusters)
        assert result == clusters
        assert any(FALLBACK_MESSAGE in r.message for r in caplog.records)

    def test_load_error_is_not_retried(self, monkeypatch):
        """A failed load is recorded once and not retried within the process."""
        monkeypatch.setattr(
            "similarity_tool.ai_refinement._dependencies_available", lambda: True
        )
        calls = {"n": 0}

        def _fail_load(self):
            # Mirror the real guard: once _load_error is set, no retry.
            if self.available or self._load_error is not None:
                return
            calls["n"] += 1
            self._load_error = "boom"

        monkeypatch.setattr(AIRefiner, "_load", _fail_load)
        refiner = AIRefiner("fake/model")
        refiner.refine([_cluster(["/a", "/b"])])
        refiner.refine([_cluster(["/a", "/b"])])
        assert calls["n"] == 1


class TestAIRefinerScope:
    def test_singleton_cluster_is_skipped(self, monkeypatch):
        """Clusters with fewer than 2 members are never sent to the model."""
        monkeypatch.setattr(
            "similarity_tool.ai_refinement._dependencies_available", lambda: True
        )
        loaded = {"n": 0}

        def _load(self):
            loaded["n"] += 1
            self._model = object()
            self._processor = object()

        monkeypatch.setattr(AIRefiner, "_load", _load)
        refiner = AIRefiner("fake/model")
        result = refiner.refine([_cluster(["/a"])])
        assert result == [_cluster(["/a"])]
        assert loaded["n"] == 0

    def test_empty_cluster_list_skips_model(self, monkeypatch):
        monkeypatch.setattr(
            "similarity_tool.ai_refinement._dependencies_available", lambda: True
        )
        loaded = {"n": 0}

        def _load(self):
            loaded["n"] += 1
            self._model = object()
            self._processor = object()

        monkeypatch.setattr(AIRefiner, "_load", _load)
        refiner = AIRefiner("fake/model")
        assert refiner.refine([]) == []
        assert loaded["n"] == 0


class TestAIRefinerEmbedding:
    def test_embedding_failure_keeps_cluster(self, monkeypatch, caplog):
        """If embedding fails, the hash-only cluster is kept and an error logged."""
        monkeypatch.setattr(
            "similarity_tool.ai_refinement._dependencies_available", lambda: True
        )

        def _load(self):
            self._model = object()
            self._processor = object()

        def _embed(self, photos):
            return None

        monkeypatch.setattr(AIRefiner, "_load", _load)
        monkeypatch.setattr(AIRefiner, "_embed", _embed)
        refiner = AIRefiner("fake/model")
        cluster = _cluster(["/a", "/b"])
        with caplog.at_level("ERROR", logger="similarity_tool.ai_refinement"):
            result = refiner.refine([cluster])
        assert result == [cluster]

    def test_embedding_splits_weak_pairs(self, monkeypatch):
        """Pairs below the similarity threshold land in separate clusters."""
        monkeypatch.setattr(
            "similarity_tool.ai_refinement._dependencies_available", lambda: True
        )

        def _load(self):
            self._model = object()
            self._processor = object()

        # Three images: a-b similar (0.95), a-c weak (0.5), b-c weak (0.5).
        # Unit vectors in R^3 with those pairwise dot products.
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.95, np.sqrt(1 - 0.95**2), 0.0])
        c = np.array([0.5, 0.08006, 0.86232])

        def _embed(self, photos):
            return _embeddings([a, b, c])

        monkeypatch.setattr(AIRefiner, "_load", _load)
        monkeypatch.setattr(AIRefiner, "_embed", _embed)
        refiner = AIRefiner("fake/model", similarity_threshold=0.85)
        cluster = _cluster(["/a", "/b", "/c"])
        result = refiner.refine([cluster])
        # a and b stay together; c is split off (singleton, so not returned).
        assert len(result) == 1
        assert {m.path for m in result[0].members} == {Path("/a"), Path("/b")}
        assert result[0].is_ai_refined
        assert result[0].ai_score is not None

    def test_all_weak_pairs_split_into_singletons(self, monkeypatch):
        """When every pair is below threshold, no refined cluster remains."""
        monkeypatch.setattr(
            "similarity_tool.ai_refinement._dependencies_available", lambda: True
        )

        def _load(self):
            self._model = object()
            self._processor = object()

        # Two orthogonal unit vectors: cosine similarity 0.0 (below threshold).
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])

        def _embed(self, photos):
            return _embeddings([a, b])

        monkeypatch.setattr(AIRefiner, "_load", _load)
        monkeypatch.setattr(AIRefiner, "_embed", _embed)
        refiner = AIRefiner("fake/model", similarity_threshold=0.85)
        result = refiner.refine([_cluster(["/a", "/b"])])
        assert result == []

    def test_refined_cluster_never_exceeds_cap(self, monkeypatch):
        """Refined clusters respect the 2x4 grid cap."""
        monkeypatch.setattr(
            "similarity_tool.ai_refinement._dependencies_available", lambda: True
        )

        def _load(self):
            self._model = object()
            self._processor = object()

        n = MAX_CLUSTER_SIZE
        # All images share the same unit vector: pairwise similarity 1.0.
        base = np.array([1.0, 0.0, 0.0, 0.0])

        def _embed(self, photos):
            return _embeddings([base] * n)

        monkeypatch.setattr(AIRefiner, "_load", _load)
        monkeypatch.setattr(AIRefiner, "_embed", _embed)
        refiner = AIRefiner("fake/model", similarity_threshold=0.85)
        cluster = _cluster([f"/img{i}" for i in range(n)])
        result = refiner.refine([cluster])
        assert all(len(c.members) <= MAX_CLUSTER_SIZE for c in result)
        assert sum(len(c.members) for c in result) == n

    def test_cluster_above_cap_is_passed_through_untouched(self, monkeypatch):
        """Clusters larger than the cap are never refined (out of scope)."""
        monkeypatch.setattr(
            "similarity_tool.ai_refinement._dependencies_available", lambda: True
        )
        loaded = {"n": 0}

        def _load(self):
            loaded["n"] += 1
            self._model = object()
            self._processor = object()

        monkeypatch.setattr(AIRefiner, "_load", _load)
        refiner = AIRefiner("fake/model", similarity_threshold=0.85)
        big = _cluster([f"/img{i}" for i in range(MAX_CLUSTER_SIZE + 1)])
        result = refiner.refine([big])
        assert result == [big]
        assert loaded["n"] == 0


class TestRefinementLabel:
    def test_hash_only_label(self):
        assert refinement_label(_cluster(["/a", "/b"])) == "hash"

    def test_ai_refined_label_contains_score(self):
        label = refinement_label(_cluster(["/a", "/b"], ai_score=0.9))
        assert label.startswith("AI ")
        assert "0.90" in label

    def test_default_threshold_matches_config_default(self):
        assert DEFAULT_AI_SIMILARITY_THRESHOLD == 0.85


class TestClusterLabel:
    def test_hash_only_label(self):
        label = cluster_label(_cluster(["/a", "/b"]), index=1)
        assert label == "Cluster 1 (2 images)"

    def test_ai_refined_label_marks_score(self):
        label = cluster_label(_cluster(["/a", "/b", "/c"], ai_score=0.92), index=3)
        assert label == "Cluster 3 (3 images) [AI 0.92]"

    def test_label_without_index(self):
        label = cluster_label(_cluster(["/a", "/b"]))
        assert label == "Cluster (2 images)"
