import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

matrices_dir = 'Writhe_Matrices_256'
csv_path = 'proteins_final.csv'
df = pd.read_csv(csv_path)

def parse_cath(code):
    parts = str(code).strip().split(".")
    if len(parts) < 4:
        return None, None, None, None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except:
        return None, None, None, None

parsed = df['cath_code'].apply(parse_cath)
df['C'] = parsed.apply(lambda x: x[0])
df['A'] = parsed.apply(lambda x: x[1])
df['T'] = parsed.apply(lambda x: x[2])
df['H'] = parsed.apply(lambda x: x[3])
df = df.dropna(subset=['C', 'A', 'T', 'H']).copy()
df['C'] = df['C'].astype(int)
df['A'] = df['A'].astype(int)
df['T'] = df['T'].astype(int)
df['H'] = df['H'].astype(int)
df = df[df['C'].isin([1, 2, 3, 4])].copy()

paths = []
C_vals = []
A_vals = []
T_vals = []
H_vals = []
missing = 0

for _, row in df.iterrows():
    fp = os.path.join(matrices_dir, f"{row['domain_id']}.npz")
    if not os.path.exists(fp):
        missing += 1
        continue
    paths.append(fp)
    C_vals.append(int(row['C']))
    A_vals.append(int(row['A']))
    T_vals.append(int(row['T']))
    H_vals.append(int(row['H']))

paths = np.array(paths)
C_vals = np.array(C_vals, dtype=np.int64)
A_vals = np.array(A_vals, dtype=np.int64)
T_vals = np.array(T_vals, dtype=np.int64)
H_vals = np.array(H_vals, dtype=np.int64)

print(f"Total proteins: {len(paths)} (missing: {missing})")


H_unique = sorted(np.unique(H_vals).tolist())
H_to_idx = {h: i for i, h in enumerate(H_unique)}
idx_to_H = {i: h for h, i in H_to_idx.items()}
H_labels = np.array([H_to_idx[h] for h in H_vals], dtype=np.int64)
num_H_classes = len(H_unique)

print(f"Number of H classes: {num_H_classes}")

label_counts = Counter(H_labels)
abundant_classes = [lbl for lbl, cnt in label_counts.items() if cnt >= 50]
medium_classes = [lbl for lbl, cnt in label_counts.items() if 10 <= cnt < 50]
rare_classes = [lbl for lbl, cnt in label_counts.items() if 2 <= cnt < 10]
singleton_classes = [lbl for lbl, cnt in label_counts.items() if cnt == 1]

print(f"Abundant classes (>=50): {len(abundant_classes)}")
print(f"Medium classes (10-49): {len(medium_classes)}")
print(f"Rare classes (2-9): {len(rare_classes)}")
print(f"Singleton classes (1): {len(singleton_classes)}")


max_samples = 6000  
min_samples = 50   

selected_indices = []
for lbl in range(num_H_classes):
    indices = np.where(H_labels == lbl)[0]
    
    if len(indices) < min_samples:
        continue
    
    if len(indices) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(indices, max_samples, replace=False)
    
    selected_indices.extend(indices.tolist())

selected_indices = np.array(selected_indices)
np.random.seed(42)
np.random.shuffle(selected_indices)

paths_sampled = paths[selected_indices]
C_vals_sampled = C_vals[selected_indices]
A_vals_sampled = A_vals[selected_indices]
T_vals_sampled = T_vals[selected_indices]
H_labels_sampled = H_labels[selected_indices]

remaining_classes = np.unique(H_labels_sampled)
print(f"\nClasses after filtering (min {min_samples} samples): {len(remaining_classes)}")
print(f"Total samples: {len(H_labels_sampled)}")

old_to_new = {old: new for new, old in enumerate(remaining_classes)}
H_labels_sampled = np.array([old_to_new[l] for l in H_labels_sampled])
idx_to_H = {new: idx_to_H[old] for old, new in old_to_new.items()}

class HierarchicalDataset(Dataset):
    def __init__(self, paths, C_vals, A_vals, T_vals, H_labels):
        self.paths = paths
        self.C_vals = C_vals
        self.A_vals = A_vals
        self.T_vals = T_vals
        self.H_labels = H_labels

    def __len__(self):
        return len(self.H_labels)

    def __getitem__(self, idx):
        mat = np.load(self.paths[idx])['matrix'].astype(np.float32)
        mat = mat[:-1, :-1]  
        m = float(mat.mean())
        s = float(mat.std()) + 1e-8
        mat = (mat - m) / s
        
        x = torch.from_numpy(mat).unsqueeze(0)
        
        
        c_val = self.C_vals[idx]
        a_val = self.A_vals[idx]
        t_val = self.T_vals[idx]
        
        c_norm = (c_val - 1) / 3.0      
        a_norm = a_val / 170.0           
        t_norm = t_val / 4090.0           
        
        c = torch.tensor([c_norm, a_norm, t_norm], dtype=torch.float32)
        y = torch.tensor(self.H_labels[idx], dtype=torch.long)
        
        return x, c, y

class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes, num_context_features=3):
        super().__init__()
        
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.context_mlp = nn.Sequential(
            nn.Linear(num_context_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
       
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, writhe_matrix, context):
        img_feat = self.features(writhe_matrix)
        img_feat = self.pool(img_feat)
        img_feat = torch.flatten(img_feat, 1)
        ctx_feat = self.context_mlp(context)
        combined = torch.cat([img_feat, ctx_feat], dim=1)
        return self.classifier(combined)


train_paths, temp_paths, train_C, temp_C, train_A, temp_A, train_T, temp_T, train_y, temp_y = train_test_split(
    paths_sampled, C_vals_sampled, A_vals_sampled, T_vals_sampled, H_labels_sampled,
    test_size=0.3, stratify=H_labels_sampled, random_state=42
)

val_paths, test_paths, val_C, test_C, val_A, test_A, val_T, test_T, val_y, test_y = train_test_split(
    temp_paths, temp_C, temp_A, temp_T, temp_y,
    test_size=0.5, stratify=temp_y, random_state=42
)

print(f"\nTrain: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

batch_size = 32


train_loader = DataLoader(
    HierarchicalDataset(train_paths, train_C, train_A, train_T, train_y),
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    prefetch_factor=4,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True
)

val_loader = DataLoader(
    HierarchicalDataset(val_paths, val_C, val_A, val_T, val_y),
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
    prefetch_factor=4,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True
)

test_loader = DataLoader(
    HierarchicalDataset(test_paths, test_C, test_A, test_T, test_y),
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
    prefetch_factor=4,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True
)


model = HierarchicalCNN(num_classes=len(remaining_classes), num_context_features=3).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")


criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

best_val_loss = float('inf')
patience = 10
patience_counter = 0
max_epochs = 200

train_losses = []
val_losses = []
val_accuracies = []

scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

print("Starting training\n")

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    epoch_start = time.time()

    for batch_idx, (matrices, context, labels_batch) in enumerate(train_loader):
        matrices = matrices.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            outputs = model(matrices, context)
            loss = criterion(outputs, labels_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        if batch_idx == 0 and epoch == 0:
            batch_time = time.time() - epoch_start
            est_epoch_time = (batch_time * len(train_loader)) / 60.0
            print(f"Estimated epoch time: {est_epoch_time:.1f} minutes\n")

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for matrices, context, labels_batch in val_loader:
            matrices = matrices.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(matrices, context)
                loss = criterion(outputs, labels_batch)
            val_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time/60:.1f}m")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss - 1e-5:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'model_state': model.state_dict(),
            'H_to_idx': H_to_idx,
            'idx_to_H': idx_to_H,
            'num_H_classes': len(remaining_classes),
            'remaining_classes': remaining_classes.tolist()
        }, "cnn_H_hierarchical_best.pth")
        print(f"Model saved (Val Loss: {best_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print("\nEvaluating on test set\n")

checkpoint = torch.load("cnn_H_hierarchical_best.pth", map_location=device)
model.load_state_dict(checkpoint['model_state'])

model.eval()
all_preds = []
all_true = []
all_C_vals = []

with torch.no_grad():
    for matrices, context, labels_batch in test_loader:
        matrices = matrices.to(device, non_blocking=True)
        context = context.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            outputs = model(matrices, context)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_true.extend(labels_batch.cpu().numpy().tolist())
        all_C_vals.extend(context[:, 0].cpu().numpy().tolist())

all_preds = np.array(all_preds)
all_true = np.array(all_true)
all_C_vals = np.array(all_C_vals)

overall_acc = (all_preds == all_true).mean()
print(f"Overall test accuracy: {overall_acc:.4f}\n")

print("C-level breakdown:")
for c_val in [0, 0.33, 0.67, 1.0]:  
    mask = np.isclose(all_C_vals, c_val, atol=0.01)
    c_preds = all_preds[mask]
    c_true = all_true[mask]
    
    if len(c_true) > 0:
        c_acc = (c_preds == c_true).mean()
        actual_c = int(c_val * 3) + 1  
        print(f"  C={actual_c}: {c_acc:.4f} (n={len(c_true)})")

print("\nClassification report:")
unique_classes = sorted(np.unique(np.concatenate([all_true, all_preds])))
target_names = [f"H{idx_to_H[i]}" for i in unique_classes]
print(classification_report(
    all_true, all_preds,
    labels=unique_classes,
    target_names=target_names,
    zero_division=0
))

cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(18, 16))
sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
plt.title(f"Confusion Matrix (H-level, {len(unique_classes)} classes)", fontsize=16)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix_H.png", dpi=200)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(range(1, len(train_losses) + 1), train_losses, linewidth=2, label='Training')
ax1.plot(range(1, len(val_losses) + 1), val_losses, linewidth=2, label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Progress')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_H.png', dpi=300)
plt.close()

print("\nTraining complete")
