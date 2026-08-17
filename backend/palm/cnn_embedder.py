"""
Lightweight CNN Palm Embedding Extractor using MobileNetV3-Small.

Provides the same interface as PalmEmbedder (embedder.py):
  - embed(aligned_bgr) -> 128-D L2-normalized float32 numpy array.

Supports both PyTorch (.pth) and ONNX Runtime (.onnx) inference.
ONNX Runtime is recommended on Raspberry Pi 4 for maximum speed (~12-18ms).
"""

import os
from typing import Optional
import cv2
import numpy as np


class PalmEmbedderCNN:
    def __init__(self, model_path: str = "mobilenet_v3_palm.onnx", embedding_dim: int = 128):
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.use_onnx = model_path.endswith(".onnx")
        self._session = None
        self._torch_model = None

        if os.path.exists(model_path):
            self.load(model_path)

    def load(self, model_path: str) -> None:
        self.model_path = model_path
        self.use_onnx = model_path.endswith(".onnx")

        if self.use_onnx:
            import onnxruntime as ort
            # Use CPU execution provider for Raspberry Pi 4 compatibility
            self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        else:
            import torch
            import torchvision.models as models
            import torch.nn as nn

            backbone = models.mobilenet_v3_small(weights=None)
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

            model = SimCLRModel(backbone, in_features, embed_dim=self.embedding_dim)
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            model.eval()
            self._torch_model = model

    @staticmethod
    def _preprocess(aligned_bgr: np.ndarray) -> np.ndarray:
        """Preprocess 224x224 BGR aligned crop for CNN input:
        1. Grayscale + CLAHE contrast enhancement for vein visibility
        2. RGB conversion + Normalization (ImageNet mean & std)
        """
        # Grayscale + CLAHE boost for NIR vein lines
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        # Resize to 224x224 if needed
        if rgb.shape[:2] != (224, 224):
            rgb = cv2.resize(rgb, (224, 224))

        # Normalize to [0, 1] then ImageNet standard
        img = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # Transpose HWC -> CHW and add Batch dimension (1, 3, 224, 224)
        tensor = np.transpose(img, (2, 0, 1))[None, ...]
        return tensor.astype(np.float32)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Takes a 224x224 aligned BGR palm image, returns 128-D L2 normalized embedding."""
        if self._session is None and self._torch_model is None:
            raise RuntimeError(f"No trained CNN model loaded from {self.model_path}. Train model using scripts/train_cnn.py first.")

        tensor = self._preprocess(aligned_bgr)

        if self.use_onnx:
            input_name = self._session.get_inputs()[0].name
            output_name = self._session.get_outputs()[0].name
            vec = self._session.run([output_name], {input_name: tensor})[0][0]
        else:
            import torch
            with torch.no_grad():
                inp = torch.from_numpy(tensor)
                vec = self._torch_model(inp)[0].numpy()

        # L2 Normalization (Unit Length Embedding)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec
