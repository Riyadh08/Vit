import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class SyntheticImageDataset(Dataset):
    def __init__(self, n: int, shape: Tuple[int, int, int], seed: int):
        c, h, w = shape
        if (c, h, w) != (3, 224, 224):
            raise ValueError("Synthetic shape must be (3, 224, 224)")
        gen = torch.Generator().manual_seed(seed)
        self.data = torch.rand((n, c, h, w), generator=gen)

    def __len__(self) -> int:
        return self.data.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class FolderImageDataset(Dataset):
    def __init__(self, root: str, max_samples: int, transform: transforms.Compose):
        self.transform = transform
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [p for p in Path(root).rglob("*") if p.suffix.lower() in allowed]
        self.files = files[:max_samples]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_calibration_loader(
    image_folder: Optional[str],
    max_samples: int,
    batch_size: int,
    shape: Tuple[int, int, int],
    seed: int,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> DataLoader:
    normalize = transforms.Normalize(mean=mean, std=std)
    folder_tf = transforms.Compose([
        transforms.Resize((shape[1], shape[2])),
        transforms.ToTensor(),
        normalize,
    ])
    if image_folder and Path(image_folder).exists():
        dataset = FolderImageDataset(image_folder, max_samples, folder_tf)
        if len(dataset) > 0:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    raw = SyntheticImageDataset(max_samples, shape, seed)
    raw.data = normalize(raw.data)
    return DataLoader(raw, batch_size=batch_size, shuffle=False)


def print_allocation(title: str, allocation: Dict[str, int], total_cost: int, budget: int) -> None:
    print(f"\n{title}")
    for name in sorted(allocation):
        print(f"  {name}: {allocation[name]}-bit")
    print(f"Total cost: {total_cost} | Budget: {budget}")


def print_omega_table(omega_map: Dict[str, Dict[int, float]]) -> None:
    print("\nOmega values per component:")
    for name in sorted(omega_map):
        pairs = ", ".join(f"{b}:{omega_map[name][b]:.6f}" for b in sorted(omega_map[name]))
        print(f"  {name} -> {pairs}")


def save_json(path: str, payload: Dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
