import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import load_vit_model
from utils import get_device, set_seed


def accuracy_counts(output: torch.Tensor, target: torch.Tensor) -> Tuple[int,int]:
    with torch.no_grad():
        _, pred5 = output.topk(5, 1, True, True)
        _, pred1 = output.topk(1, 1, True, True)
        pred5 = pred5.cpu()
        pred1 = pred1.squeeze(1).cpu()
        target_cpu = target.cpu()
        top1 = (pred1 == target_cpu).sum().item()
        top5 = sum(1 for i in range(pred5.size(0)) if target_cpu[i].item() in pred5[i].tolist())
        return top1, top5


def build_loader(val_dir: str, batch_size: int, num_workers: int):
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    ds = datasets.ImageFolder(val_dir, transform=val_transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", type=str, default=r"D:\\imagenet_full\\val")
    parser.add_argument("--model-name", type=str, default="deit_tiny_patch16_224")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    print(f"Loading data from: {args.val_dir}")
    loader = build_loader(args.val_dir, args.batch_size, args.num_workers)
    print(f"Dataset size: {len(loader.dataset)} samples, {len(loader)} batches (loader built)")

    print(f"Loading model: {args.model_name}")
    model = load_vit_model(args.model_name, device)
    model.eval()

    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    start_all = time.time()
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(loader):
            if batch_idx >= args.num_batches:
                break
            t0 = time.time()
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            top1_c, top5_c = accuracy_counts(outputs, targets)

            batch_size = images.size(0)
            total_top1 += top1_c
            total_top5 += top5_c
            total_samples += batch_size

            t1 = time.time()
            print(f"Batch {batch_idx+1}/{args.num_batches} | samples: {batch_size} | batch_time: {t1-t0:.3f}s | batch_top1: {top1_c}/{batch_size} | batch_top5: {top5_c}/{batch_size}")

    end_all = time.time()
    top1_acc = (total_top1 / total_samples) * 100 if total_samples else 0.0
    top5_acc = (total_top5 / total_samples) * 100 if total_samples else 0.0

    print("\n=== Subset Validation Summary ===")
    print(f"Batches processed: {min(args.num_batches, len(loader))}")
    print(f"Samples evaluated: {total_samples}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Total time: {end_all-start_all:.2f}s")

if __name__ == '__main__':
    main()
