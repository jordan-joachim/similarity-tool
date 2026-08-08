"""Group similar images from perceptual-hash distances.

Two images are considered similar when their pHash Hamming distance is within
``phash_threshold`` AND their dHash Hamming distance is within
``dhash_threshold``. The module builds an undirected graph over those pairs,
extracts connected components, and splits any component larger than the 2x4
grid size (8) into smaller clusters whose members are still pairwise within
both thresholds.

The pairwise comparison is vectorized with NumPy: hashes are decoded into
``uint64`` arrays and the Hamming distance matrix is computed with
``numpy.bitwise_count``, so a month with thousands of images produces a
compact boolean matrix instead of a Python object per pair.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from similarity_tool.models import Cluster, HashRecord, PhotoFile

log = logging.getLogger(__name__)

#: Maximum number of members in a displayed cluster (the 2x4 thumbnail grid).
MAX_CLUSTER_SIZE = 8

#: Message shown when a scan finds no similar-image clusters.
EMPTY_STATE_MESSAGE = "No similar images found"


@dataclass
class ClusterResult:
    """The outcome of clustering a month's hash records.

    ``clusters`` holds the displayable clusters (each with 2..=``max_size``
    members). ``empty_message`` is the empty-state text the GUI should show
    when no clusters were found.
    """

    clusters: list[Cluster] = field(default_factory=list)
    empty_message: str = EMPTY_STATE_MESSAGE

    @property
    def is_empty(self) -> bool:
        return not self.clusters


def _decode_hashes(records: Sequence[HashRecord]) -> np.ndarray:
    """Decode hex-encoded hashes into a ``(n, 2)`` uint64 array.

    Malformed or empty hashes decode to ``0``; the caller must mask those
    records out of the adjacency matrix so they never match anything.
    """
    n = len(records)
    values = np.zeros((n, 2), dtype=np.uint64)
    for i, record in enumerate(records):
        for col, value in ((0, record.phash), (1, record.dhash)):
            if value:
                try:
                    values[i, col] = int(value, 16)
                except ValueError:
                    # Malformed hash: leave as 0 and mark it invalid below.
                    values[i, col] = 0
    return values


def _valid_mask(records: Sequence[HashRecord]) -> np.ndarray:
    """Return a boolean mask of records with well-formed hashes of equal length."""
    mask = np.ones(len(records), dtype=bool)
    for i, record in enumerate(records):
        for value in (record.phash, record.dhash):
            if not value:
                mask[i] = False
                break
            try:
                int(value, 16)
            except ValueError:
                mask[i] = False
                break
    return mask


def within_thresholds(
    records: Sequence[HashRecord],
    phash_threshold: int,
    dhash_threshold: int,
) -> np.ndarray:
    """Return a boolean ``n x n`` matrix of pairs within both thresholds.

    The diagonal is always ``False`` (an image is not similar to itself).
    Records with empty or malformed hashes never match anything.
    """
    n = len(records)
    if n == 0:
        return np.zeros((0, 0), dtype=bool)
    values = _decode_hashes(records)
    phash = values[:, 0]
    dhash = values[:, 1]
    phash_dist = np.bitwise_count(phash[:, None] ^ phash[None, :])
    dhash_dist = np.bitwise_count(dhash[:, None] ^ dhash[None, :])
    within = (phash_dist <= phash_threshold) & (dhash_dist <= dhash_threshold)
    np.fill_diagonal(within, False)
    within &= _valid_mask(records)[:, None] & _valid_mask(records)[None, :]
    return within


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    """Return the connected components of the graph as lists of indices."""
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    components: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        component: list[int] = []
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = np.flatnonzero(adjacency[node] & ~seen)
            for neighbor in neighbors:
                seen[neighbor] = True
                stack.append(int(neighbor))
        components.append(component)
    return components


def split_component(
    component: list[int], adjacency: np.ndarray, max_size: int
) -> list[list[int]]:
    """Split a connected component into cliques of at most *max_size* members.

    Every member of the component is assigned to exactly one clique, and every
    pair within a clique satisfies the adjacency relation (i.e. is within both
    hash thresholds). The greedy expansion is deterministic: seeds are chosen
    by descending degree (ties broken by index), and candidates are added in
    index order.
    """
    if len(component) <= max_size:
        # Fast path: a component that is already a clique (every pair within
        # both thresholds) is kept whole. Non-clique components must be
        # partitioned even when small, because a connected component can
        # contain pairs that violate the thresholds.
        sub = adjacency[component][:, component]
        if sub.sum() == len(component) * (len(component) - 1):
            return [component]

    degrees = adjacency[component][:, component].sum(axis=1)
    order = sorted(range(len(component)), key=lambda i: (-int(degrees[i]), i))
    assigned = np.zeros(len(component), dtype=bool)
    cliques: list[list[int]] = []

    for seed in order:
        if assigned[seed]:
            continue
        clique = [component[seed]]
        assigned[seed] = True
        # Candidates are all unassigned members adjacent to every current member.
        candidates = np.flatnonzero(~assigned)
        while len(clique) < max_size:
            candidate = None
            for idx in candidates:
                node = component[int(idx)]
                if all(adjacency[node, member] for member in clique):
                    candidate = int(idx)
                    break
            if candidate is None:
                break
            clique.append(component[candidate])
            assigned[candidate] = True
            candidates = np.flatnonzero(~assigned)
        cliques.append(clique)

    return cliques


def build_clusters(
    records: Sequence[HashRecord],
    photos: Sequence[PhotoFile],
    phash_threshold: int,
    dhash_threshold: int,
    max_size: int = MAX_CLUSTER_SIZE,
    hash_algorithms: Sequence[str] | None = None,
) -> ClusterResult:
    """Group *photos* into clusters of images within both hash thresholds.

    *records* and *photos* are parallel lists: each record's ``photo_path``
    must match the corresponding photo's path. Photos without a record (e.g.
    corrupt images skipped by the hasher) and records without a photo are
    ignored. Clusters are capped at *max_size* members; larger connected
    components are split into smaller clusters whose members remain pairwise
    within both thresholds. Clusters are returned largest-first, with members
    sorted by file size (largest first) for a stable display order.
    """
    if not records or not photos:
        return ClusterResult()

    photo_by_path = {str(photo.path): photo for photo in photos}
    adjacency = within_thresholds(records, phash_threshold, dhash_threshold)

    clusters: list[Cluster] = []
    for component in connected_components(adjacency):
        if len(component) < 2:
            continue
        for clique in split_component(component, adjacency, max_size):
            if len(clique) < 2:
                continue
            members = [photo_by_path.get(records[i].photo_path) for i in clique]
            members = [m for m in members if m is not None]
            if len(members) < 2:
                continue
            members.sort(key=lambda m: m.size, reverse=True)
            clusters.append(
                Cluster(
                    members=members,
                    hash_algorithms=list(hash_algorithms) if hash_algorithms else [],
                )
            )

    clusters.sort(key=lambda c: len(c.members), reverse=True)
    return ClusterResult(clusters=clusters)
