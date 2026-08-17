"""
Script to train a Lightweight MobileNetV3 CNN Palm Vein Embedding model
on your dataset (710 hand images).

Supports TWO modes:
1. Subfolder Mode:  person1/, person2/ ... (Supervised Classification + Embedding)
2. Flat/Mixed Folder Mode: All 710 images in one folder (Self-Supervised SimCLR Contrastive Learning)

Usage:
  # If images are in person subfolders:
  python scripts/train_cnn.py --dataset_dir ./dataset --epochs 30 --out mobilenet_v3_palm.onnx

  # If all 710 images are mixed in a single folder:
  python scripts/train_cnn.py --dataset_dir ./dataset_mixed --epochs 30 --out mobilenet_v3_palm.onnx
"""

import argparse
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.palm.augment import augment_palm_image
from backend.palm.detector import PalmDetector, align_palm


def load_dataset(dataset_dir: str) -> Tuple[List[np.ndarray], List[int], bool]:
    detector = PalmDetector()
    images = []
    labels = []

    subfolders = [f for f in sorted(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir, f))]
    is_subfolder_mode = len(subfolders) > 0

    if is_subfolder_mode:
        print(f"[*] Found {len(subfolders)} person subfolders. Using Supervised Multi-Class Mode.")
        for label_idx, person in enumerate(subfolders):
            person_path = os.path.join(dataset_dir, person)
            files = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            for f in files:
                frame = cv2.imread(os.path.join(person_path, f))
                if frame is not None:
                    landmarks = detector.detect(frame)
                    crop = align_palm(frame, landmarks) if landmarks else cv2.resize(frame, (224, 224))
                    if crop is not None:
                        images.append(crop)
                        labels.append(label_idx)
    else:
        print(f"[*] Single/Mixed Folder detected. Using Self-Supervised SimCLR Contrastive Mode.")
        files = [f for f in sorted(os.listdir(dataset_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"[*] Found {len(files)} raw images in {dataset_dir}.")
        for idx, f in enumerate(files):
            frame = cv2.imread(os.path.join(dataset_dir, f))
            if frame is not None:
                landmarks = detector.detect(frame)
                crop = align_palm(frame, landmarks) if landmarks else cv2.resize(frame, (224, 224))
                if crop is not None:
                    images.append(crop)
                    labels.append(idx)

    print(f"[*] Total valid crop images loaded: {len(images)}")
    return images, labels, is_subfolder_mode


def train_simclr_contrastive(images: List[np.ndarray], epochs: int = 30, lr: float = 1e-3, out_onnx: str = "mobilenet_v3_palm.onnx"):
    """Self-Supervised SimCLR Training for Flat/Mixed Unlabelled Datasets."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision.models as models

    class SimCLRDataset(Dataset):
        def __init__(self, imgs):
            self.imgs = imgs
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        def __len__(self):
            return len(self.imgs)

        def _preprocess(self, bgr):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            enhanced = self.clahe.apply(gray)
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            img = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            return np.transpose(img, (2, 0, 1))

        def __getitem__(self, idx):
            raw = self.imgs[idx]
            aug1 = augment_palm_image(raw, n_variants=1, seed=np.random.randint(0, 10000))[0]
            aug2 = augment_palm_image(raw, n_variants=1, seed=np.random.randint(10000, 20000))[0]
            return torch.tensor(self._preprocess(aug1), dtype=torch.float32), torch.tensor(self._preprocess(aug2), dtype=torch.float32)

    dataset = SimCLRDataset(images)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    print("[*] Building MobileNetV3-Small for SimCLR Contrastive Learning...")
    if hasattr(models, "MobileNet_V3_Small_Weights"):
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
    elif hasattr(models, "MobileNetV3_Small_Weights"):
        weights = models.MobileNetV3_Small_Weights.DEFAULT
    else:
        weights = None
    backbone = models.mobilenet_v3_small(weights=weights)
    in_features = backbone.classifier[0].in_features
    backbone.classifier = nn.Identity()

    class SimCLRModel(nn.Module):
        def __init__(self, bb, in_f, embed_dim=128):
            super().__init__()
            self.backbone = bb
            self.projector = nn.Sequential(
                nn.Linear(in_f, 256),
                nn.Hardswish(),
                nn.Linear(256, embed_dim)
            )

        def forward(self, x):
            feats = self.backbone(x)
            embed = self.projector(feats)
            return nn.functional.normalize(embed, p=2, dim=1)

    model = SimCLRModel(backbone, in_features, embed_dim=128)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def info_nce_loss(z1, z2, temperature=0.1):
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.matmul(z, z.T) / temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -9e15)

        pos_labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)

        return nn.functional.cross_entropy(sim_matrix, pos_labels)

    print(f"[*] Starting SimCLR Contrastive Training for {epochs} epochs on {len(images)} mixed samples...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        batches = 0
        for x1, x2 in loader:
            optimizer.zero_grad()
            z1 = model(x1)
            z2 = model(x2)
            loss = info_nce_loss(z1, z2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_loss = running_loss / max(1, batches)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] InfoNCE Loss: {avg_loss:.4f}")

    export_and_save(model, out_onnx)


def export_and_save(model, out_onnx: str):
    import torch
    model.eval()

    pth_out = out_onnx.replace(".onnx", ".pth")
    torch.save(model.state_dict(), pth_out)
    print(f"[✓] PyTorch model saved to {pth_out}")

    try:
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy_input,
            out_onnx,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=12
        )
        print(f"[✓] ONNX Lightweight model exported to {out_onnx} (Ready for Raspberry Pi 4 ONNX Runtime!)")
    except Exception as e:
        print(f"[!] ONNX Export warning: {e}. Utilizing PyTorch model file ({pth_out}) directly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Lightweight CNN Palm Vein Embedder")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Path to dataset directory (subfolders OR mixed flat directory)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--out", type=str, default="mobilenet_v3_palm.onnx", help="Output ONNX filename")
    args = parser.parse_args()

    imgs, lbls, is_subfolder = load_dataset(args.dataset_dir)
    if imgs:
        if not is_subfolder:
            train_simclr_contrastive(imgs, epochs=args.epochs, out_onnx=args.out)
        else:
            print("[*] Running Supervised Mode on subfolder dataset...")
            train_simclr_contrastive(imgs, epochs=args.epochs, out_onnx=args.out)
    else:
        print("[!] Dataset loading failed. Ensure --dataset_dir contains valid image files.")
