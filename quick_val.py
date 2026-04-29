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


# Setup
print("Setup...", flush=True)
sys.stdout.flush()
set_seed(42)
device = get_device()
print(f"Device: {device}", flush=True)

# Build validation loader
print("Building loader...", flush=True)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
val_dataset = datasets.ImageFolder(r"D:\imagenet_full\val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

print(f"Dataset size: {len(val_dataset)}", flush=True)

# Load model
print("Loading model...", flush=True)
model = load_vit_model("deit_tiny_patch16_224", device)

# Validate on subset
print("Starting validation...", flush=True)
model.eval()
top1_total = 0.0
top5_total = 0.0
num_samples = 0

with torch.no_grad():
    for batch_idx, (images, target) in enumerate(val_loader):
        print(f"Batch {batch_idx+1}", flush=True)
        sys.stdout.flush()
        
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
        
        print(f"  Samples processed: {num_samples}", flush=True)
        sys.stdout.flush()

top1_acc = top1_total / num_samples
top5_acc = top5_total / num_samples

print(f"\nTop-1 Accuracy: {top1_acc:.2f}%", flush=True)
print(f"Top-5 Accuracy: {top5_acc:.2f}%", flush=True)
