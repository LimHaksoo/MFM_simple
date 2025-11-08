# model/llm_decoder/model.py
# -*- coding: utf-8 -*-
"""
LLaMA Decoder with Vision Prefix
- inputs_embeds로 vision prefix 토큰을 선행한 뒤 텍스트 임베딩을 이어붙여 학습/생성
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# JISU ADDED
# 메모리 절약: 4bit 양자화(+auto offload) 시도 → 실패 시 bf16 + auto offload
# 선택: bitsandbytes가 있으면 4bit로, 없으면 자동 폴백
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False
class LlamaDecoderWithVisionPrefix(nn.Module):
    def __init__(self, pretrained_name: str, device: Optional[torch.device] = None, freeze_lm: bool = True):
        super().__init__()

        load_kwargs = dict()
        # 1) 가능한 경우 4bit 양자화 + auto offload (GPU+CPU 분산)
        if _HAS_BNB:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            load_kwargs.update(
                quantization_config=bnb_cfg,
                device_map="auto",              # 중요: 장치 자동 분산, .to(device) 금지
                low_cpu_mem_usage=True,
            )
        else:
            # 2) 폴백: bf16(가능 시) + auto offload
            load_kwargs.update(
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                device_map="auto",              # auto offload
                low_cpu_mem_usage=True,
            )

        self.llm = AutoModelForCausalLM.from_pretrained(pretrained_name, **load_kwargs)

        # device_map="auto"로 로드했을 때는 .to(device) 삭제
        # if device is not None:  
        #     self.llm.to(device)

        # 학습용 메모리 절약
        self.llm.config.use_cache = False           # grad 계산 시 캐시 끔
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()

        if freeze_lm:
            for p in self.llm.parameters():
                p.requires_grad = False

        self.hidden_size = self.llm.config.hidden_size
        self.token_embed = self.llm.get_input_embeddings()

        # 토크나이저 유틸 (pad 토큰 설정)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

# class LlamaDecoderWithVisionPrefix(nn.Module):
#     def __init__(self, pretrained_name: str, device: Optional[torch.device] = None, freeze_lm: bool = True):
#         super().__init__()
#         self.llm = AutoModelForCausalLM.from_pretrained(pretrained_name)
#         if freeze_lm:
#             for p in self.llm.parameters():
#                 p.requires_grad = False
#         if device is not None:
#             self.llm.to(device)
#         self.hidden_size = self.llm.config.hidden_size
#         self.token_embed = self.llm.get_input_embeddings()

#         # 토크나이저 유틸 (pad 토큰 설정)
#         self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

    def _concat_prefix(
        self,
        vision_prefix: torch.Tensor,      # (B, V, H)
        input_ids: Optional[torch.Tensor] = None,  # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
    ):
        B, V, H = vision_prefix.shape
        if input_ids is not None:
            text_emb = self.token_embed(input_ids)  # (B, T, H)
            inputs_embeds = torch.cat([vision_prefix, text_emb], dim=1)  # (B, V+T, H)
            if attention_mask is None:
                attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            prefix_mask = torch.ones((B, V), dtype=attention_mask.dtype, device=attention_mask.device)
            attn = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, V+T)
        else:
            # pure prefix-only (rare)
            inputs_embeds = vision_prefix
            attn = torch.ones((B, V), dtype=torch.long, device=vision_prefix.device)
        return inputs_embeds, attn

    def forward(
        self,
        vision_prefix: torch.Tensor,          # (B, V, H)
        input_ids: Optional[torch.Tensor],    # (B, T)
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None # (B, T)
    ) -> Dict[str, Any]:
        """
        Training (teacher forcing) 용도:
        - labels가 주어지면 loss 반환
        - vision_prefix 길이(V)에 해당하는 라벨은 -100으로 가립니다.
        """
        inputs_embeds, attn = self._concat_prefix(vision_prefix, input_ids, attention_mask)

        if labels is not None:
            # vision prefix 부분은 로스 제외
            B, V, H = vision_prefix.shape
            ignore = torch.full((labels.size(0), V), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([ignore, labels], dim=1)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels
        )
        return {"loss": out.loss if hasattr(out, "loss") else None, "logits": out.logits}

    @torch.no_grad()
    def generate(
        self,
        vision_prefix: torch.Tensor,          # (B, V, H)
        input_ids: Optional[torch.Tensor],    # (B, T) prompt
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        **gen_kwargs
    ):
        inputs_embeds, attn = self._concat_prefix(vision_prefix, input_ids, attention_mask)
        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            **gen_kwargs
        )
