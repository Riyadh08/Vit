#!/usr/bin/env python3
"""Minimal ImageNet validation test."""

import torch
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms

# Import project modules
from models import load_vit_model
from utils import get_device, set_seed

def main():
    # Setup
    set_seed(42)
    device = get_device()
    print(f"Device: {device}\n")
    
    # Load model
    print("Loading model...")
    model = load_vit_model("deit_tiny_patch16_224", device)
    model.eval()
    print("✓ Model loaded\n")
    
    # Image transform
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Get ImageNet val directory
    val_dir = Path(r"D:\imagenet_full\val")
    if not val_dir.exists():
        print(f"ERROR: {val_dir} does not exist")
        return
    
    # Get class directories
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes")
    
    # Collect all images
    print("\nCollecting images...")
    all_images = []
    for class_idx, class_dir in enumerate(class_dirs):
        images = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg"))
        for img_path in images:
            all_images.append((img_path, class_idx))
    
    print(f"Found {len(all_images)} total images")
    
    # Validation
    print(f"\nValidating on {len(all_images)} images...\n")
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for idx, (img_path, label_idx) in enumerate(all_images):
            try:
                # Load and transform image
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Inference
                output = model(img_tensor)
                
                # Get predictions
                _, pred_top5 = output.topk(5, 1, True, True)
                _, pred_top1 = output.topk(1, 1, True, True)
                
                # Check correctness
                if pred_top1[0][0].item() == label_idx:
                    top1_correct += 1
                    
                if label_idx in pred_top5[0].tolist():
                    top5_correct += 1
                
                total += 1
                
                # Print progress
                if (idx + 1) % 5000 == 0:
                    print(f"Processed: {idx + 1}/{len(all_images)} | "
                          f"Top-1: {100*top1_correct/total:.2f}% | "
                          f"Top-5: {100*top5_correct/total:.2f}%")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Final results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Model: deit_tiny_patch16_224 (pretrained)")
    print(f"Dataset: ImageNet Validation")
    print(f"Images Evaluated: {total}")
    print(f"\nAccuracy:")
    print(f"  Top-1: {100*top1_correct/total:.2f}%")
    print(f"  Top-5: {100*top5_correct/total:.2f}%")
    print("="*60)
    
    # Save results
    with open("validation_results.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("IMAGENET VALIDATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Model: deit_tiny_patch16_224 (pretrained)\n")
        f.write(f"Dataset: ImageNet Validation\n")
        f.write(f"Images Evaluated: {total}\n\n")
        f.write(f"Top-1 Accuracy: {100*top1_correct/total:.2f}%\n")
        f.write(f"Top-5 Accuracy: {100*top5_correct/total:.2f}%\n")
        f.write("="*60 + "\n")
    
    print(f"\n✓ Results saved to validation_results.txt")

if __name__ == "__main__":
    main()
