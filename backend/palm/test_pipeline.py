"""
Sanity-check the alignment math, embedder, and matcher without needing a
webcam or the downloaded MediaPipe model file. Run with:

    python -m backend.palm.test_pipeline
"""

import numpy as np

from backend.palm.detector import Landmark, align_palm
from backend.palm.embedder import PalmEmbedder
from backend.palm.matcher import PalmMatcher


def _mock_landmarks(cx=112, cy=140, tilt_deg=0.0, size=1.0, img_w=224, img_h=224):
    """Build a fake 21-point hand landmark set with wrist/MCPs at a chosen
    center, tilt and scale, so we can test align_palm() geometry."""
    ang = np.radians(tilt_deg)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

    base = {
        0: np.array([0, 60]),     # wrist
        5: np.array([-35, -10]),  # index MCP
        9: np.array([0, -40]),    # middle MCP
        17: np.array([35, -10]),  # pinky MCP
    }
    pts = [np.array([0.0, 0.0])] * 21
    for idx, offset in base.items():
        p = (rot @ (offset * size)) + np.array([cx, cy])
        pts[idx] = p
    # fill remaining indices with something plausible so len() checks pass
    for i in range(21):
        if i not in base:
            pts[i] = np.array([cx, cy])

    return [Landmark(x=float(p[0]) / img_w, y=float(p[1]) / img_h) for p in pts]


def _synthetic_palm(seed: int, size: int = 224) -> np.ndarray:
    """Deterministic pseudo-palm texture per 'person', plus small per-photo
    noise, standing in for real photographs during this dry run."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    # smooth it a bit so HOG has coherent gradients, not pure noise
    import cv2
    base = cv2.GaussianBlur(base, (15, 15), 0)
    return base


def _synthetic_photo(seed: int, photo_variation: int, size: int = 224) -> np.ndarray:
    img = _synthetic_palm(seed, size).astype(np.int16)
    rng = np.random.default_rng(seed * 1000 + photo_variation)
    noise = rng.integers(-10, 10, size=img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img


def test_align_palm_geometry():
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    lms_straight = _mock_landmarks(cx=112, cy=140, tilt_deg=0.0)
    aligned = align_palm(frame, lms_straight, out_size=224)
    assert aligned is not None
    assert aligned.shape == (224, 224, 3)

    lms_tilted = _mock_landmarks(cx=90, cy=160, tilt_deg=35.0, size=1.3)
    aligned_tilted = align_palm(frame, lms_tilted, out_size=224)
    assert aligned_tilted is not None
    assert aligned_tilted.shape == (224, 224, 3)
    print("[PASS] align_palm handles centered and tilted/scaled hands")


def test_embed_and_match():
    n_people = 5
    photos_per_person = 30  # >= embedding_dim requirement for PCA fit

    enrollment_imgs, enrollment_labels = [], []
    for person_id in range(n_people):
        for photo_i in range(photos_per_person):
            enrollment_imgs.append(_synthetic_photo(seed=person_id, photo_variation=photo_i))
            enrollment_labels.append(person_id)

    embedder = PalmEmbedder(embedding_dim=32)
    embedder.fit(enrollment_imgs)

    matcher = PalmMatcher(match_threshold=0.5)
    for img, label in zip(enrollment_imgs, enrollment_labels):
        matcher.add(customer_id=label, embedding=embedder.embed(img))

    # A brand-new photo of person 2's palm should match person 2
    query_img = _synthetic_photo(seed=2, photo_variation=999)
    query_emb = embedder.embed(query_img)
    matched_id, score = matcher.identify(query_emb)
    print(f"[INFO] query matched customer_id={matched_id} score={score:.3f}")
    assert matched_id == 2, f"expected customer 2, got {matched_id}"

    # verify() should confirm the SAME person on the second (payment) scan
    ok, vscore = matcher.verify(customer_id=2, embedding=query_emb)
    assert ok, f"verify() should confirm same-person match, score={vscore:.3f}"

    # verify() against a DIFFERENT registered customer_id should fail
    ok_wrong, vscore_wrong = matcher.verify(customer_id=4, embedding=query_emb)
    assert not ok_wrong, f"verify() incorrectly confirmed wrong customer, score={vscore_wrong:.3f}"

    print("[PASS] embed + identify + verify correctly distinguish 5 synthetic palms")


if __name__ == "__main__":
    test_align_palm_geometry()
    test_embed_and_match()
    print("\nAll self-tests passed.")
