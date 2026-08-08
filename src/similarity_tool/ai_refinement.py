"""Optional CPU-only CLIP/SigLIP second-stage refinement for hash clusters.

When enabled, this module loads a small CLIP model (via ``transformers`` +
``torch``) on first use, embeds the members of each hash cluster, and splits
any pair whose cosine similarity falls below ``ai_similarity_threshold`` into
separate clusters. The model always runs on the CPU with float32; no CUDA
device is ever requested. If ``transformers``/``torch`` are missing or the
model cannot be loaded, the module logs an error and returns the hash-only
clusters unchanged, so the scan still produces usable results.

The module imports neither ``transformers`` nor ``torch`` at import time:
``ai_refinement=false`` never triggers a model load or download because the
caller gates on the config flag before invoking :func:`refine_clusters`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from similarity_tool.clusters import (
    MAX_CLUSTER_SIZE,
    connected_components,
    split_component,
)
from similarity_tool.models import Cluster

log = logging.getLogger(__name__)

#: Minimum cosine similarity kept by the AI stage (matches the config default).
DEFAULT_AI_SIMILARITY_THRESHOLD = 0.85

#: Message logged when the AI stage cannot run and hash-only clusters are used.
FALLBACK_MESSAGE = "AI refinement unavailable; using hash-only clusters."


def _dependencies_available() -> bool:
    """Return True when ``transformers`` and ``torch`` can be imported."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


class AIRefiner:
    """Lazy, CPU-only CLIP/SigLIP refiner for hash clusters.

    The model is loaded on first use (in :meth:`refine`), not at construction,
    so a disabled or never-used AI stage never downloads anything. All
    computation runs on the CPU with float32; no CUDA device is requested.
    """

    def __init__(
        self,
        model_name: str,
        similarity_threshold: float = DEFAULT_AI_SIMILARITY_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self._model = None
        self._processor = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        """True when the model and processor are loaded and usable."""
        return self._model is not None and self._processor is not None

    def _load(self) -> None:
        """Load the CLIP model and processor on the CPU (first use only).

        A failed load is recorded in ``_load_error`` and logged; subsequent
        calls do not retry the download within the same process.
        """
        if self.available or self._load_error is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            self._load_error = f"transformers/torch not installed ({exc})"
            log.error("AI refinement unavailable: %s", self._load_error)
            return
        try:
            log.info(
                "Loading AI refinement model %s on CPU (first use; downloads if not cached)...",
                self.model_name,
            )
            self._model = CLIPModel.from_pretrained(
                self.model_name, torch_dtype=torch.float32
            )
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to("cpu")
        except Exception as exc:  # noqa: BLE001 - any model failure falls back
            self._load_error = str(exc)
            log.error("Could not load AI refinement model %s: %s", self.model_name, exc)
            self._model = None
            self._processor = None

    def _embed(self, photos: Sequence[Cluster]) -> np.ndarray | None:
        """Return an ``(n, d)`` float32 matrix of L2-normalized embeddings.

        Returns ``None`` (and logs) if the images cannot be preprocessed or
        embedded, in which case the caller keeps the hash-only cluster. Any
        model runtime error is caught here so a broken or mismatched model
        falls back to hash-only clusters instead of crashing the scan.
        """
        import torch

        images = [str(photo.path) for photo in photos]
        try:
            inputs = self._processor(images=images, return_tensors="pt")
            with torch.no_grad():
                output = self._model.get_image_features(**inputs)
        except Exception as exc:  # noqa: BLE001 - any model failure falls back
            log.error("AI refinement could not embed images: %s", exc)
            return None
        # transformers >= 5 returns a BaseModelOutputWithPooling object;
        # older versions return the pooled tensor directly.
        features = getattr(output, "pooler_output", output)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def refine(self, clusters: Sequence[Cluster]) -> list[Cluster]:
        """Refine *clusters* with CLIP similarity, splitting weak pairs.

        Only clusters with 2..=``MAX_CLUSTER_SIZE`` members are processed;
        larger or singleton clusters are returned unchanged. If the
        dependencies or the model are unavailable, the original clusters are
        returned unchanged and an error is logged (hash-only fallback).
        """
        if not clusters:
            return list(clusters)
        if not _dependencies_available():
            log.error(
                "AI refinement unavailable: transformers/torch not installed. %s",
                FALLBACK_MESSAGE,
            )
            return list(clusters)
        # Only clusters of 2..=MAX_CLUSTER_SIZE members are refined; if none
        # qualify, the model is never loaded (no download, no CPU work).
        scoped = [c for c in clusters if 2 <= len(c) <= MAX_CLUSTER_SIZE]
        if not scoped:
            return list(clusters)
        self._load()
        if not self.available:
            log.error("%s", FALLBACK_MESSAGE)
            return list(clusters)

        refined: list[Cluster] = []
        for cluster in clusters:
            if len(cluster) < 2 or len(cluster) > MAX_CLUSTER_SIZE:
                refined.append(cluster)
                continue
            embeddings = self._embed(cluster.members)
            if embeddings is None:
                refined.append(cluster)
                continue
            similarity = embeddings @ embeddings.T
            adjacency = similarity >= self.similarity_threshold
            np.fill_diagonal(adjacency, False)
            # Reuse the deterministic clique splitter so refined clusters keep
            # the same invariants as hash clusters: every pair within a
            # cluster is above the threshold and no cluster exceeds the cap.
            for component in connected_components(adjacency):
                if len(component) < 2:
                    continue
                for clique in split_component(component, adjacency, MAX_CLUSTER_SIZE):
                    if len(clique) < 2:
                        continue
                    members = [cluster.members[i] for i in clique]
                    members.sort(key=lambda m: m.size, reverse=True)
                    sub = similarity[np.ix_(clique, clique)]
                    upper = sub[np.triu_indices(len(clique), 1)]
                    min_sim = float(np.min(upper)) if upper.size else 1.0
                    refined.append(
                        Cluster(
                            members=members,
                            hash_algorithms=list(cluster.hash_algorithms),
                            ai_score=min_sim,
                        )
                    )
        return refined


def refine_clusters(clusters: Sequence[Cluster], cfg) -> list[Cluster]:
    """Refine *clusters* when ``cfg.ai_refinement`` is enabled.

    When AI refinement is disabled (the default), the clusters are returned
    unchanged without importing ``transformers``/``torch`` or touching the
    model. When enabled, a CPU-only refiner is used and any failure falls back
    to the hash-only clusters. *cfg* needs ``ai_refinement``, ``ai_model`` and
    ``ai_similarity_threshold`` attributes (the :class:`Config` dataclass
    provides them).
    """
    if not cfg.ai_refinement:
        return list(clusters)
    refiner = AIRefiner(cfg.ai_model, cfg.ai_similarity_threshold)
    return refiner.refine(clusters)


def refinement_label(cluster: Cluster) -> str:
    """Return a short label distinguishing AI-refined from hash-only clusters.

    AI-refined clusters carry their minimum pairwise similarity score; hash-only
    clusters get a plain label. The GUI uses this to satisfy the requirement
    that the UI distinguishes the two kinds of clusters.
    """
    if cluster.is_ai_refined:
        return f"AI {cluster.ai_score:.2f}"
    return "hash"


def cluster_label(cluster: Cluster, index: int | None = None) -> str:
    """Return a display label for a cluster, marking AI-refined clusters.

    Hash-only clusters are labeled ``Cluster N (M images)``; AI-refined
    clusters append their minimum pairwise similarity score, e.g.
    ``Cluster 1 (4 images) [AI 0.92]``. The GUI result list and grid header
    use this label so the user can tell the two kinds of clusters apart.
    """
    prefix = f"Cluster {index}" if index is not None else "Cluster"
    if cluster.is_ai_refined:
        return f"{prefix} ({len(cluster.members)} images) [AI {cluster.ai_score:.2f}]"
    return f"{prefix} ({len(cluster.members)} images)"
