"""
Palm embedding extractor.

Turns an aligned palm image (output of detector.align_palm) into a fixed
length vector, such that:
  - two photos of the SAME palm  -> vectors close together
  - photos of DIFFERENT palms    -> vectors far apart

Baseline approach (this file): Histogram of Oriented Gradients (HOG) on
the palm's principal-line/texture pattern, reduced with PCA. This is a
classical, well-established palm-print recognition technique, it is
lightweight (CPU only, no GPU/heavy deep-learning dependency), and it is
fully explainable in a viva -- you can point at exactly which gradients
drove a match.

Upgrade path for later (and a natural ablation-study angle for a paper):
swap this for a CNN embedding fine-tuned on palm images with a triplet
or ArcFace loss, and compare False-Accept/False-Reject rates against
this HOG+PCA baseline.
"""

from typing import List, Optional

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA


class PalmEmbedder:
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self._pca: Optional[PCA] = None

    # ---- feature extraction -------------------------------------------------

    @staticmethod
    def _raw_hog_features(aligned_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # normalize lighting differences
        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return features.astype(np.float32)

    # ---- fitting the PCA projection -----------------------------------------

    def fit(self, aligned_images: List[np.ndarray]) -> None:
        """Fit the PCA projection once, offline, on a batch of enrollment
        images (ideally: several palms x several photos each). Must be
        called once before `embed()` and the fitted PCA persisted
        (see save/load) so every deployment uses the same projection."""
        if len(aligned_images) < self.embedding_dim:
            raise ValueError(
                f"Need at least {self.embedding_dim} enrollment images to fit "
                f"a {self.embedding_dim}-D PCA; got {len(aligned_images)}."
            )
        raw = np.stack([self._raw_hog_features(img) for img in aligned_images])
        self._pca = PCA(n_components=self.embedding_dim, whiten=True)
        self._pca.fit(raw)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Call fit() (or load a saved PCA) before embed().")
        raw = self._raw_hog_features(aligned_bgr).reshape(1, -1)
        vec = self._pca.transform(raw)[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ---- persistence ----------------------------------------------------------

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self._pca, path)

    def load(self, path: str) -> None:
        import joblib
        self._pca = joblib.load(path)
