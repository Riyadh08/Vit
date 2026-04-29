from dataclasses import dataclass
from typing import List, Tuple

import timm
import torch
import torch.nn as nn


@dataclass
class ComponentSpec:
    name: str
    block_idx: int
    kind: str
    module_path: str
    module: nn.Module
    num_params: int


def load_vit_model(model_name: str, device: torch.device) -> nn.Module:
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(device)
    return model


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def extract_vit_components(model: nn.Module) -> List[ComponentSpec]:
    components: List[ComponentSpec] = []
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Model does not expose transformer blocks via .blocks")
    for i, block in enumerate(blocks):
        entries: List[Tuple[str, nn.Module, str]] = [
            ("qkv", block.attn.qkv, f"blocks.{i}.attn.qkv"),
            ("proj", block.attn.proj, f"blocks.{i}.attn.proj"),
            ("fc1", block.mlp.fc1, f"blocks.{i}.mlp.fc1"),
            ("fc2", block.mlp.fc2, f"blocks.{i}.mlp.fc2"),
        ]
        for kind, mod, path in entries:
            name = f"block{i}.{kind}"
            components.append(ComponentSpec(name, i, kind, path, mod, _count_params(mod)))
    return components
