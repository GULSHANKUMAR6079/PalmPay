"""
Synthesize plausible variations of a single palm photo when real photo
count per person is very low (e.g. one shot per hand).

IMPORTANT LIMITATION: augmentation can only jitter around what's already
in the one real photo (its lighting, its exact pose) -- it cannot invent
a genuinely new viewing angle or a different day's lighting condition.
It's enough to get the pipeline running and to stop the matcher relying
on a single static template, but it is NOT a substitute for real photo
diversity. Before reporting FAR/FRR numbers for a paper, collect a
proper dataset: multiple people, multiple photos each, ideally across
more than one session/day (see README "Dataset" section).
"""

from typing import List

import cv2
import numpy as np


def augment_palm_image(aligned_bgr: np.ndarray, n_variants: int = 15, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = aligned_bgr.shape[:2]
    variants = []

    for _ in range(n_variants):
        img = aligned_bgr.copy()

        # small rotation (hand isn't held at the exact same angle twice)
        angle = rng.uniform(-12, 12)
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, rot, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # small translation
        tx, ty = rng.uniform(-8, 8, size=2)
        trans = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, trans, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # brightness / contrast jitter (different terminal lighting)
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.uniform(-15, 15)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # mild sensor-noise jitter
        noise = rng.normal(0, 4, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        variants.append(img)

    return variants
