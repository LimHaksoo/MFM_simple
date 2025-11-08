# model/img_encoder/model.py
# -*- coding: utf-8 -*-
"""
CLIP 기반 Vision Encoder
- open_clip_torch 우선 사용
- 미설치 시 openai/clip로 폴백
출력: (B, vision_dim) 전역 임베딩
주의: LLaVA는 ViT 패치 토큰을 쓰지만, 간단화를 위해 전역 임베딩을 Projector가 num_vision_tokens로 확장합니다.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

class VisionEncoderCLIP(nn.Module):
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        freeze: bool = True,
        normalize: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.normalize = normalize
        self.device = device
        self.backend = None

        self.model, self.vision_dim = self._load_model(model_name, pretrained)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
        if device is not None:
            self.model.to(device)

    def _load_model(self, model_name: str, pretrained: str):
        # 1) open_clip_torch
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            vision_dim = model.visual.output_dim  # e.g., 768 or 1024
            self.backend = "open_clip"
            return model, vision_dim
        except Exception:
            pass
        # 2) openai/clip
        try:
            import clip
            model, _ = clip.load(model_name, jit=False)
            # OpenAI CLIP에서는 proj 이후 512 차원(대부분)으로 반환됨
            # 시그니처 상 내부 visual.width 가능하지만 안전하게 512로 가정
            vision_dim = 512
            self.backend = "clip"
            return model, vision_dim
        except Exception as e:
            raise RuntimeError(
                f"Neither open_clip_torch nor openai/clip is available. "
                f"Install one of them. Original error: {repr(e)}"
            )

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B,3,H,W) CLIP 정규화/리사이즈가 이미 적용된 텐서
        Returns:
            feats: (B, vision_dim) L2 정규화 선택
        """
        if self.backend == "open_clip":
            feats = self.model.encode_image(images)
        elif self.backend == "clip":
            feats = self.model.encode_image(images)
        else:
            raise RuntimeError("Invalid backend")
        if self.normalize:
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
        return feats

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)

    def get_output_dim(self) -> int:
        return self.vision_dim
