import json
import time
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# Try to import modern torchvision weights, fall back to older API if needed
try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    HAS_WEIGHTS_API = True
except ImportError:
    from torchvision.models import efficientnet_b0
    HAS_WEIGHTS_API = False

def train_model():
    # 1. GPU Support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))
    
    # Define paths
    base_dir = Path("dataset")
    train_dir = base_dir / "train"
    val_dir = base_dir / "validation"
    
    # 2. Data Augmentation and Preprocessing
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Datasets and Dataloaders
    if not train_dir.exists() or not val_dir.exists():
        print("Error: Train or validation directories do not exist. Please run dataset split script first.")
        return
        
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transforms)
    
    train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
    )

    val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
    )
    
    print(f"Classes: {train_dataset.classes}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    targets = train_dataset.targets
    class_counts = Counter(targets)

    print("\nClass distribution:")
    for i, cls in enumerate(train_dataset.classes):
       print(f"{cls}: {class_counts[i]}")

    # 4. Handle Class Imbalance with Weighted CrossEntropyLoss
    targets = train_dataset.targets
    class_counts = Counter(targets)
    num_classes = len(train_dataset.classes)
    
    # Safe computation of class weights ordered by class index
    class_samples = [class_counts[i] for i in range(num_classes)]
    total_samples = sum(class_samples)
    class_weights = [total_samples / (num_classes * count) if count > 0 else 1.0 for count in class_samples]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # 5. Initialize Model (EfficientNet-B0 Transfer Learning)
    if HAS_WEIGHTS_API:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    else:
        model = efficientnet_b0(pretrained=True)
        
    # Freeze everything
    for param in model.parameters():
      param.requires_grad = False

# Replace final classifier layer
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

# Unfreeze last EfficientNet block
    for param in model.features[-1].parameters():
      param.requires_grad = True

# Unfreeze classifier
    for param in model.classifier.parameters():
      param.requires_grad = True
        
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    
    # 6. Optimizer (Adam on classifier parameters only)
    optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-5,
    weight_decay=1e-4
   )
    
    # Track statistics for plotting
    epochs = 25
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 7. Training Loop
    best_acc = 0
    print("\nStarting Training...")
    for epoch in range(epochs):
        start = time.time()

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(
                model.state_dict(),
                models_dir / "emstrap_best.pth"
            )
            print(f"Best model saved: {best_acc:.2f}%")

        # Save metrics for plotting
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.2f}%")
        print(f"Time: {time.time()-start:.1f}s")
        print("-" * 50)
        
    # 8. Save Model and Metadata
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    torch.save(model.state_dict(),
           models_dir/"emstrap_final.pth")
    print(f"Saved model state dictionary to {models_dir / 'emstrap_final.pth'}")
    
    # Save classes
    with open(models_dir / "classes.json", "w") as f:
        json.dump(train_dataset.classes, f, indent=4)
    print(f"Saved class index metadata to {models_dir / 'classes.json'}")
    
    # 9. Plot training and validation accuracy curves
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accs, label="Train Accuracy", marker='o')
    plt.plot(range(1, epochs + 1), val_accs, label="Validation Accuracy", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy Curves")
    plt.legend()
    plt.grid(True)
    
    plot_path = reports_dir / "accuracy_curves.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved accuracy plot to {plot_path}")

if __name__ == "__main__":
    train_model()

