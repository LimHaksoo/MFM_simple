# model/llm_encoder/model.py
# -*- coding: utf-8 -*-
"""
LLM Encoder (텍스트를 LLM 은닉공간으로 임베딩)
+ Vision->LLM Projector (간단한 2층 MLP로 비전 전역 임베딩을 num_vision_tokens 만큼 확장)
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
# from transformers import AutoTokenizer, AutoModel
from transformers import AutoConfig, AutoTokenizer

def build_tokenizer(pretrained_name: str, use_fast: bool = True):
    tok = AutoTokenizer.from_pretrained(pretrained_name, use_fast=use_fast)
    # LLaMA는 pad 토큰이 없을 수 있음 -> 안전하게 추가
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# class LlamaEncoder(nn.Module):
#     def __init__(self, pretrained_name: str, device: Optional[torch.device] = None, freeze: bool = True):
#         super().__init__()
#         self.model = AutoModel.from_pretrained(pretrained_name)
#         if freeze:
#             for p in self.model.parameters():
#                 p.requires_grad = False
#         if device is not None:
#             self.model.to(device)
#         self.hidden_size = self.model.config.hidden_size

#     @torch.no_grad()
#     def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Returns:
#             hidden_states: (B, T, hidden_size)
#         """
#         out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
#         return out.last_hidden_state'

# JISU ADDED
class LlamaEncoder(nn.Module):
    """Hidden size만 가져오는 경량 래퍼 (모델 가중치 로드 안 함)."""
    def __init__(self, pretrained_name: str, device: Optional[str] = None, freeze: bool = True):
        super().__init__()
        cfg = AutoConfig.from_pretrained(pretrained_name)
        self.hidden_size = int(cfg.hidden_size)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        raise RuntimeError("Config-holder only")
    
class MultiModalProjector(nn.Module):
    """
    비전 전역 임베딩 (B, vision_dim) -> (B, num_vision_tokens, llm_dim)
    간단한 2층 MLP + LayerNorm (LLaVA 스타일 근사)
    """
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_vision_tokens: int = 32,
        hidden_dim: Optional[int] = None,
        use_ln: bool = True
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(vision_dim, llm_dim)
        self.num_vision_tokens = num_vision_tokens
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_vision_tokens * llm_dim),
        )
        self.ln = nn.LayerNorm(llm_dim) if use_ln else nn.Identity()
        self.llm_dim = llm_dim

    def forward(self, vision_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_feat: (B, vision_dim) - 전역 CLIP 임베딩
        Returns:
            vision_tokens: (B, num_vision_tokens, llm_dim) - LLM 입력 프리픽스 토큰
        """
        B = vision_feat.size(0)
        x = self.proj(vision_feat)  # (B, num_tokens * llm_dim)
        x = x.view(B, self.num_vision_tokens, self.llm_dim)
        x = self.ln(x)
        return x
