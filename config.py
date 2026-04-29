from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ExperimentConfig:
    model_name: str = "deit_tiny_patch16_224"
    bit_choices: List[int] = field(default_factory=lambda: [2, 3, 4])
    temperature: float = 1.0
    batch_size: int = 16
    max_calib_samples: int = 128
    synthetic_shape: Tuple[int, int, int] = (3, 224, 224)
    calib_image_folder: Optional[str] = None
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    budget_mode: str = "avg_bit"
    avg_bit_target: int = 3
    fixed_budget: Optional[int] = None
    dp_scale_factor: int = 1000
    topk_refine: int = 8
    synthetic_seed: int = 7
    print_per_component_omega: bool = True
    save_allocation_json_path: str = "allocation.json"


def resolve_output_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()
