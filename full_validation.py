import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import load_vit_model
from utils import get_device, set_seed
import time

set_seed(42)
device = get_device()
print(f"Device: {device}", flush=True)

# Load data
print("Loading dataset...", flush=True)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
val_dataset = datasets.ImageFolder(r'D:\imagenet_full\val', transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
print(f"Dataset loaded: {len(val_dataset)} samples", flush=True)

# Load model
print("Loading model...", flush=True)
model = load_vit_model('deit_tiny_patch16_224', device)
model.eval()
print("Model loaded", flush=True)

# Test one batch
print("\nTesting one batch...", flush=True)
start = time.time()
images, targets = next(iter(val_loader))
images = images.to(device)
targets = targets.to(device)
print(f"Input shape: {images.shape}", flush=True)

print("Running forward pass...", flush=True)
with torch.no_grad():
    output = model(images)
print(f"Output shape: {output.shape}", flush=True)

# Calculate accuracy
_, pred = output.topk(5, 1, True, True)
correct = pred.eq(targets.unsqueeze(1))
acc = correct[:, 0].float().mean() * 100
end = time.time()
print(f"Batch Top-1 Accuracy: {acc:.2f}%", flush=True)
print(f"Time for batch: {end-start:.2f}s", flush=True)

# Now run full validation
print("\n" + "="*50, flush=True)
print("Starting Full Validation...", flush=True)
print("="*50, flush=True)

top1_total = 0.0
top5_total = 0.0
num_samples = 0
batch_times = []

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        batch_start = time.time()
        
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        output = model(images)
        
        # Calculate accuracy
        _, pred_top5 = output.topk(5, 1, True, True)
        _, pred_top1 = output.topk(1, 1, True, True)
        
        correct_top5 = pred_top5.eq(targets.unsqueeze(1)).any(dim=1)
        correct_top1 = pred_top1.squeeze(1).eq(targets)
        
        acc_top1 = correct_top1.float().sum().item()
        acc_top5 = correct_top5.float().sum().item()
        
        batch_size = images.size(0)
        top1_total += acc_top1
        top5_total += acc_top5
        num_samples += batch_size
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if (batch_idx + 1) % 50 == 0:
            avg_time = sum(batch_times) / len(batch_times)
            print(f"Batch {batch_idx+1}/{len(val_loader)} | Samples: {num_samples} | Avg batch time: {avg_time:.3f}s", flush=True)

top1_acc = (top1_total / num_samples) * 100
top5_acc = (top5_total / num_samples) * 100

print("\n" + "="*50, flush=True)
print("FINAL RESULTS:", flush=True)
print(f"  Top-1 Accuracy: {top1_acc:.2f}%", flush=True)
print(f"  Top-5 Accuracy: {top5_acc:.2f}%", flush=True)
print(f"  Total Samples: {num_samples}", flush=True)
print("="*50, flush=True)

# Save results
with open("validation_results.txt", "w") as f:
    f.write("="*50 + "\n")
    f.write("VALIDATION RESULTS\n")
    f.write("="*50 + "\n")
    f.write(f"Model: deit_tiny_patch16_224\n")
    f.write(f"Validation Set: ImageNet Validation (50,000 images)\n")
    f.write(f"Batch Size: 256\n")
    f.write(f"\nResults:\n")
    f.write(f"  Top-1 Accuracy: {top1_acc:.2f}%\n")
    f.write(f"  Top-5 Accuracy: {top5_acc:.2f}%\n")
    f.write(f"  Total Samples: {num_samples}\n")
    f.write("="*50 + "\n")

print("\nResults saved to validation_results.txt", flush=True)
