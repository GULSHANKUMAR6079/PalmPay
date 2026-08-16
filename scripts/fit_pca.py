"""
Run this ONCE, before starting the API for the first time.

The PalmEmbedder's PCA projection has to be fit on a batch of real palm
photos before it can produce meaningful embeddings for anyone. Point
this script at a folder of photos, and it will save `pca.joblib` for
main.py to load.

Works with very few real photos (e.g. one per teammate's hand) by
generating augmented variants of each -- see backend/palm/augment.py for
why that's a stopgap, not a substitute for a real multi-photo dataset.

Usage:
    python scripts/fit_pca.py --photos_dir ./bootstrap_photos --out ./pca.joblib
"""

import argparse
import glob
import os
import sys

# Ensure project root is in sys.path so 'backend' package can be imported directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2

from backend.palm.augment import augment_palm_image
from backend.palm.detector import PalmDetector, align_palm
from backend.palm.embedder import PalmEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos_dir", required=True, help="Folder of raw palm photos (.jpg/.png)")
    parser.add_argument("--model_path", default="hand_landmarker.task")
    parser.add_argument("--out", default="./pca.joblib")
    parser.add_argument("--embedding_dim", type=int, default=24,
                         help="Keep this LOW when you have few real photos -- a high-dimensional "
                              "PCA fit on a handful of images (even augmented) mostly memorizes "
                              "noise instead of learning generalizable palm structure.")
    parser.add_argument("--augment_per_image", type=int, default=20,
                         help="Synthetic variants generated per real photo to pad out the fit set.")
    args = parser.parse_args()

    detector = PalmDetector(model_path=args.model_path)
    embedder = PalmEmbedder(embedding_dim=args.embedding_dim)

    real_aligned = []
    paths = sorted(glob.glob(os.path.join(args.photos_dir, "*")))
    
    total_found = len(paths)
    count_unread = 0
    count_no_hand = 0
    count_no_align = 0

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"[skip] could not read {path}")
            count_unread += 1
            continue
        landmarks = detector.detect(frame)
        if landmarks is None:
            print(f"[skip] no hand found in {path}")
            count_no_hand += 1
            continue
        aligned = align_palm(frame, landmarks)
        if aligned is None:
            print(f"[skip] could not align {path}")
            count_no_align += 1
            continue
        real_aligned.append(aligned)

    count_success = len(real_aligned)
    count_failed = count_unread + count_no_hand + count_no_align
    fail_rate = (count_failed / total_found * 100.0) if total_found > 0 else 0.0

    print("\n" + "=" * 55)
    print("DATASET PROCESSING SUMMARY")
    print("=" * 55)
    print(f"Total images found             : {total_found}")
    print(f"Successfully detected & aligned : {count_success} ({100.0 - fail_rate:.1f}%)")
    print(f"Failed processing              : {count_failed} ({fail_rate:.1f}%)")
    print(f"  - Could not read image file  : {count_unread}")
    print(f"  - No hand landmarks detected : {count_no_hand}")
    print(f"  - Could not align palm crop  : {count_no_align}")
    print("=" * 55)

    if fail_rate > 30.0:
        print(
            f"\n[WARNING] High detection failure rate ({fail_rate:.1f}% > 30%).\n"
            "          This dataset may not be well-suited for palm-print recognition\n"
            "          (e.g., non-hand images, severe occlusion, extreme angles, or blurry photos)."
        )

    if count_success < 15:
        print(
            "\n  [!] Note: Very few usable real photos. PCA below will be fit mostly on\n"
            "      augmented copies, which is fine to verify the pipeline end-to-end,\n"
            "      but not sufficient for evaluating paper-grade accuracy metrics (FAR/FRR)."
        )
    if count_success == 0:
        raise SystemExit("\nNo usable photos found -- nothing to fit on.")

    fit_set = list(real_aligned)
    for img in real_aligned:
        fit_set.extend(augment_palm_image(img, n_variants=args.augment_per_image))

    print(f"\nFitting {args.embedding_dim}-D PCA on {len(fit_set)} images "
          f"({count_success} real + {len(fit_set) - count_success} augmented)...")
    embedder.fit(fit_set)
    embedder.save(args.out)
    print(f"Saved PCA projection to {args.out}")


if __name__ == "__main__":
    main()
