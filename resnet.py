# ==============================================================================
# 1. SETUP AND DATASET DOWNLOAD
# ==============================================================================
import os
import copy
from collections import Counter
import numpy as np

# ML and Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# PyTorch
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

import kagglehub

# ==============================================================================
# 2. HYPERPARAMETERS & CONFIGURATION
# ==============================================================================
# We define ROOT_DATA_DIR as a global variable, but we will set its exact path
# inside the main block after downloading/locating the Kaggle dataset.
ROOT_DATA_DIR = None 

MODEL_NAME = "resnet50"     # The core deep learning architecture.
BATCH_SIZE = 16        # Number of images processed at once.
NUM_EPOCHS_FROZEN = 10       # <-- Set to 2 for ablation study
NUM_EPOCHS_FINETUNE = 20     # <-- Set to 3 for ablation study
INIT_LR = 1e-4              # Initial learning rate.
FT_LR = 1e-5                # Smaller learning rate for fine-tuning.
NUM_WORKERS = 1             # CPU threads used for loading data.
IMG_SIZE = 224              # ResNet-50 requires 224x224 input images.
VAL_SPLIT = 0.20            # 20% of data used for validation.
CHECKPOINT = "resnet_rice_best.pth" # File name to save the best model weights.
SEED = 42                   # Fixed seed ensures results are reproducible.

# Set the seed for random operations to guarantee exact identical runs
torch.manual_seed(SEED)
np.random.seed(SEED)

# Automatically use the GPU if available, otherwise use CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 3. DATA PREPROCESSING (TRANSFORMS)
# ==============================================================================
# Current Phase: Experiment 1 (Baseline - Resize Only)
# --------- Transforms (Experiment 2: + Normalization) ---------
# --------- Transforms (Experiment 3: + Light Augmentation) ---------
# --------- Transforms (Experiment 4: Full Pipeline) ---------
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ==============================================================================
# 5. MODEL ARCHITECTURE (Defined here, called in main)
# ==============================================================================
def build_resnet(name="resnet50", num_classes=6, pretrained=True):
    # Load the pre-trained ResNet-50 model
    model = models.resnet50(pretrained=pretrained)
    
    # Replace the final classification layer
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model

# ==============================================================================
# 6. TRAINING LOOP (Defined here, called in main)
# ==============================================================================

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)

    W = x.size()[2]
    H = x.size()[3]
    
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y, y[index], lam

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def run_epoch(loader, model, criterion, optimizer=None, train=True, scaler=None):
    """Handles one full pass over the dataset (training or validation)."""
    model.train() if train else model.eval()
    running_loss, running_corrects, n_samples = 0.0, 0, 0
    
    loop = tqdm(loader, leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        if train:
            optimizer.zero_grad()

        # --- CUTMIX LOGIC ---
        # Apply CutMix only during training with a 50% probability
        apply_cutmix = train and np.random.rand() < 0.5
        if apply_cutmix:
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels)
            
        with torch.set_grad_enabled(train):
            # If using mixed precision for faster training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    
                    # Calculate loss differently depending on if CutMix was applied
                    if apply_cutmix:
                        loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = criterion(outputs, labels)
                        
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            
            # If training normally without mixed precision
            else:
                outputs = model(inputs)
                
                # Calculate loss differently depending on if CutMix was applied
                if apply_cutmix:
                    loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    loss = criterion(outputs, labels)
                    
                if train:
                    loss.backward()
                    optimizer.step()
            
            preds = torch.argmax(outputs, dim=1)
            
        # --- METRICS TRACKING ---
        running_loss += loss.item() * inputs.size(0)
        
        # When CutMix is applied, images have two labels. 
        # We check accuracy against targets_a (the dominant original label) for train metrics.
        if apply_cutmix:
            running_corrects += torch.sum(preds == targets_a).item()
        else:
            running_corrects += torch.sum(preds == labels).item()
            
        n_samples += inputs.size(0)
        loop.set_description(f"{'train' if train else 'val'} loss: {loss.item():.4f}")
        
    return running_loss / n_samples, running_corrects / n_samples


# ==============================================================================
# MAIN EXECUTION BLOCK (Required for Windows Multiprocessing)
# ==============================================================================
if __name__ == '__main__':
    print("Using device:", DEVICE)
    
    # POINT DIRECTLY TO YOUR NEW 12,000 IMAGE DATASET
    ROOT_DATA_DIR = r"D:\RESNET MODEL\Rice_Leaf_AUG"
    print(f"Dataset located at: {ROOT_DATA_DIR}")
    
    # Check if directory exists
    if not os.path.isdir(ROOT_DATA_DIR):
        raise FileNotFoundError(f"Root data dir not found: {ROOT_DATA_DIR}")

    # Load datasets
    full_ds_train_tf = datasets.ImageFolder(ROOT_DATA_DIR, transform=train_tf)
    full_ds_val_tf = datasets.ImageFolder(ROOT_DATA_DIR, transform=val_tf) 

    class_names = full_ds_train_tf.classes
    num_classes = len(class_names)
    print("Classes (detected):", class_names)

    all_indices = list(range(len(full_ds_train_tf)))
    all_labels = [y for _, y in full_ds_train_tf.imgs]

    train_idx, val_idx = train_test_split(
        all_indices, test_size=VAL_SPLIT, stratify=all_labels, random_state=SEED
    )

    train_ds = Subset(full_ds_train_tf, train_idx)
    val_ds   = Subset(full_ds_val_tf, val_idx)

    train_labels = [all_labels[i] for i in train_idx]
    train_class_counts = Counter(train_labels)
    class_weights_for_sampler = {cls: 1.0 / cnt for cls, cnt in train_class_counts.items()}
    sample_weights = [class_weights_for_sampler[label] for label in train_labels]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Initialize DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Initialize Model, Criterion, and Scaler
    model = build_resnet(MODEL_NAME, num_classes=num_classes, pretrained=True).to(DEVICE)

    weights_for_loss = torch.tensor(
        [1.0 / train_class_counts[i] if train_class_counts[i] > 0 else 1.0 for i in range(num_classes)],
        dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights_for_loss)

    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    # --- PHASE 1: Train only the new final layer ---
    print("\nPhase 1: Training classifier head only")
    for p in model.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=INIT_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_FROZEN + NUM_EPOCHS_FINETUNE)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS_FROZEN):
        train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer, train=True, scaler=scaler)
        val_loss, val_acc = run_epoch(val_loader, model, criterion, train=False, scaler=scaler)
        print(f"[Phase1] Epoch {epoch+1}/{NUM_EPOCHS_FROZEN} - train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save({"model_state": best_wts, "classes": class_names}, CHECKPOINT)
        scheduler.step()

    # --- PHASE 2: Unfreeze layer4 and fine-tune ---
    print("\nPhase 2: Fine-tuning layer4 and fc")
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("layer4") or name.startswith("fc")

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=FT_LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS_FINETUNE)

    for epoch in range(NUM_EPOCHS_FINETUNE):
        train_loss, train_acc = run_epoch(train_loader, model, criterion, optimizer, train=True, scaler=scaler)
        val_loss, val_acc = run_epoch(val_loader, model, criterion, train=False, scaler=scaler)
        print(f"[FT] Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE} - train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save({"model_state": best_wts, "classes": class_names}, CHECKPOINT)
        scheduler.step()

    # --- FINAL EVALUATION ---
    model.load_state_dict(best_wts)
    model.eval()
    all_preds, all_labels = [], []
    
    print("\nRunning final evaluation on best weights...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Final eval"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # ==============================================================================
    # 8. GENERATE PUBLICATION-READY GRAPHS
    # ==============================================================================
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import precision_recall_fscore_support

    print("\nGenerating graphs for the paper...")

    # 1. Plot and Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - ResNet50 (30 Epochs, CutMix)', fontsize=14)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300) # dpi=300 is required for IEEE papers
    plt.close()

    # 2. Plot and Save Metrics Bar Chart
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, labels=range(num_classes))
    
    x = np.arange(num_classes)
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label='Precision', color='#1f77b4')
    plt.bar(x, recall, width, label='Recall', color='#ff7f0e')
    plt.bar(x + width, f1, width, label='F1-Score', color='#2ca02c')

    plt.title('Model Performance Metrics by Disease Class', fontsize=14)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend(loc='lower right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('metrics_barchart.png', dpi=300)
    plt.close()

    print("Graphs saved successfully as 'confusion_matrix.png' and 'metrics_barchart.png'!")