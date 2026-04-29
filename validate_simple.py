import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import load_vit_model
from utils import get_device, set_seed


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> list:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            res.append(acc)
        return res


def build_imagenet_val_loader(
    val_dir: str,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Build ImageNet validation dataloader"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return val_loader


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model on validation set"""
    model.eval()
    
    top1_total = 0.0
    top5_total = 0.0
    num_samples = 0
    
    with torch.no_grad():
        total_batches = len(val_loader)
        for batch_idx, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(images)
            
            # Compute accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            
            batch_size = images.size(0)
            top1_total += acc1[0].item() * batch_size
            top5_total += acc5[0].item() * batch_size
            num_samples += batch_size
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                msg = f"  Progress: {batch_idx + 1}/{total_batches} batches processed ({num_samples} samples)"
                print(msg, flush=True)
                sys.stdout.flush()
    
    top1_acc = top1_total / num_samples
    top5_acc = top5_total / num_samples
    
    return top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(description="Validate ViT on ImageNet")
    parser.add_argument(
        "--val-dir",
        type=str,
        default=r"D:\imagenet_full\val",
        help="Path to ImageNet validation set",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deit_tiny_patch16_224",
        help="Model name to load from timm",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="validation_results.txt",
        help="File to save validation results",
    )
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}", flush=True)
    
    # Check if validation directory exists
    val_path = Path(args.val_dir)
    if not val_path.exists():
        print(f"Error: Validation directory not found: {args.val_dir}")
        return
    
    print(f"\nLoading ImageNet validation set from: {args.val_dir}", flush=True)
    
    # Build validation loader
    val_loader = build_imagenet_val_loader(
        args.val_dir,
        args.batch_size,
        args.num_workers,
    )
    print(f"Number of validation samples: {len(val_loader.dataset)}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    
    # Load model
    print(f"\nLoading pretrained model: {args.model_name}", flush=True)
    model = load_vit_model(args.model_name, device)
    
    # Validate
    print("\nRunning validation...", flush=True)
    top1_acc, top5_acc = validate(model, val_loader, device)
    
    # Print results
    result_str = "\n" + "=" * 50 + "\n"
    result_str += "Validation Results:\n"
    result_str += f"  Top-1 Accuracy: {top1_acc:.2f}%\n"
    result_str += f"  Top-5 Accuracy: {top5_acc:.2f}%\n"
    result_str += "=" * 50
    
    print(result_str, flush=True)
    
    # Save to file
    with open(args.output_file, "w") as f:
        f.write(result_str)
    print(f"\nResults saved to: {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
