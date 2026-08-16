"""
Identity matching over enrolled palm embeddings.

Deliberately a plain in-memory numpy nearest-neighbour search rather than
a vector DB (FAISS/Qdrant/etc.) -- a payment terminal serving hundreds or
low thousands of enrolled customers doesn't need approximate search
infra, and keeping this dependency-free makes the prototype easier to
run. If you scale this to a large customer base later, swap `_search`
for a FAISS IndexFlatIP without touching the rest of the pipeline.

The `threshold` is the single most safety-critical number in this whole
project: too low and a stranger's palm gets matched to someone else's
bank account (False Accept -- the dangerous failure mode for a payment
system); too high and legitimate customers keep getting rejected (False
Reject -- just annoying). Tune it empirically against a labelled
val set and report both FAR and FRR in your evaluation, don't guess.
"""

import threading
from typing import List, Optional, Tuple

import numpy as np


class PalmMatcher:
    def __init__(self, match_threshold: float = 0.85):
        self.threshold = match_threshold
        self._embeddings: List[np.ndarray] = []
        self._customer_ids: List[int] = []
        self._lock = threading.Lock()

    def add(self, customer_id: int, embedding: np.ndarray) -> None:
        norm_emb = embedding / (np.linalg.norm(embedding) or 1.0)
        with self._lock:
            self._embeddings.append(norm_emb)
            self._customer_ids.append(customer_id)

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """Returns (customer_id, similarity) for the best match, or
        (None, best_similarity_seen) if nothing cleared the threshold --
        callers MUST treat None as 'reject', never fall back to a guess."""
        with self._lock:
            if not self._embeddings:
                return None, 0.0
            embeddings_matrix = np.stack(self._embeddings)
            customer_ids = list(self._customer_ids)

        query = embedding / (np.linalg.norm(embedding) or 1.0)
        sims = embeddings_matrix @ query  # cosine sim (unit vectors)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < self.threshold:
            return None, best_score
        return customer_ids[best_idx], best_score

    def verify(self, customer_id: int, embedding: np.ndarray) -> Tuple[bool, float]:
        """Used for the SECOND palm scan (payment authorization step):
        confirms the new scan still matches the SAME customer_id that was
        identified at session start, rather than just finding the best
        match across everyone. This stops a palm-swap between step 1 and
        step 2 from authorizing someone else's payment."""
        with self._lock:
            embeddings_list = list(self._embeddings)
            customer_ids = list(self._customer_ids)

        query = embedding / (np.linalg.norm(embedding) or 1.0)
        best_score = -1.0
        for cid, emb in zip(customer_ids, embeddings_list):
            if cid == customer_id:
                score = float(emb @ query)
                best_score = max(best_score, score)
        return best_score >= self.threshold, best_score
