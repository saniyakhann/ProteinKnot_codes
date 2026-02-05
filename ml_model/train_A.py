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

matrices_dir = 'Writhe_Matrices_256'
csv_path = 'proteins_final.csv'
df = pd.read_csv(csv_path)

def parse_cath(code):
    parts = str(code).strip().split(".")
    if len(parts) < 2:
        return None, None
    try:
        return int(parts[0]), int(parts[1])
    except:
        return None, None

parsed = df['cath_code'].apply(parse_cath)
df['C'] = parsed.apply(lambda x: x[0])
df['A'] = parsed.apply(lambda x: x[1])
df = df.dropna(subset=['C', 'A']).copy()
df['C'] = df['C'].astype(int)
df['A'] = df['A'].astype(int)
df = df[df['C'].isin([1, 2, 3, 4])].copy()

paths = []
C_vals = []
A_vals = []
missing = 0

for _, row in df.iterrows():
    fp = os.path.join(matrices_dir, f"{row['domain_id']}.npz")
    if not os.path.exists(fp):
        missing += 1
        continue
    paths.append(fp)
    C_vals.append(int(row['C']))
    A_vals.append(int(row['A']))

paths = np.array(paths)
C_vals = np.array(C_vals, dtype=np.int64)
A_vals = np.array(A_vals, dtype=np.int64)

print(f"Total proteins: {len(paths)} (missing: {missing})")

A_unique = sorted(np.unique(A_vals).tolist())
A_to_idx = {a: i for i, a in enumerate(A_unique)}
idx_to_A = {i: a for a, i in A_to_idx.items()}
A_labels = np.array([A_to_idx[a] for a in A_vals], dtype=np.int64)
num_A_classes = len(A_unique)

print(f"Number of A classes: {num_A_classes}")

label_counts = Counter(A_labels)
abundant_classes = [lbl for lbl, cnt in label_counts.items() if cnt >= 50]
medium_classes = [lbl for lbl, cnt in label_counts.items() if 10 <= cnt < 50]
rare_classes = [lbl for lbl, cnt in label_counts.items() if 2 <= cnt < 10]
singleton_classes = [lbl for lbl, cnt in label_counts.items() if cnt == 1]

max_sampled = 500
min_required_per_class = 7

selected_indices = []
for lbl in range(num_A_classes):
    indices = np.where(A_labels == lbl)[0]

    if len(indices) > max_sampled:
        np.random.seed(42)
        indices = np.random.choice(indices, max_sampled, replace=False)

    if len(indices) < min_required_per_class:
        np.random.seed(42)
        indices = np.random.choice(indices, min_required_per_class, replace=True)

    selected_indices.extend(indices.tolist())

selected_indices = np.array(selected_indices)
np.random.seed(42)
np.random.shuffle(selected_indices)

paths_sampled = paths[selected_indices]
C_vals_sampled = C_vals[selected_indices]
A_labels_sampled = A_labels[selected_indices]

class HierarchicalDataset(Dataset):
    def __init__(self, paths, C_vals, A_labels):
        self.paths = paths
        self.C_vals = C_vals
        self.A_labels = A_labels

    def __len__(self):
        return len(self.A_labels)

    def __getitem__(self, idx):
        mat = np.load(self.paths[idx])['matrix'].astype(np.float32)
        m = float(mat.mean())
        s = float(mat.std()) + 1e-8
        mat = (mat - m) / s

        x = torch.from_numpy(mat).unsqueeze(0)
        c = torch.tensor([self.C_vals[idx]], dtype=torch.float32)
        y = torch.tensor(self.A_labels[idx], dtype=torch.long)

        return x, c, y

class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes, num_context_features=1):
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

train_paths, temp_paths, train_C, temp_C, train_y, temp_y = train_test_split(
    paths_sampled, C_vals_sampled, A_labels_sampled,
    test_size=0.3, stratify=A_labels_sampled, random_state=42
)

val_paths, test_paths, val_C, test_C, val_y, test_y = train_test_split(
    temp_paths, temp_C, temp_y,
    test_size=0.5, stratify=temp_y, random_state=42
)

print(f"\nTrain: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

batch_size = 32

train_class_counts = np.bincount(train_y, minlength=num_A_classes).astype(np.float32)
class_weights = (train_class_counts.sum() / (train_class_counts + 1e-8))
class_weights = class_weights / class_weights.mean()
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

train_loader = DataLoader(
    HierarchicalDataset(train_paths, train_C, train_y),
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    prefetch_factor=4,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True
)

val_loader = DataLoader(
    HierarchicalDataset(val_paths, val_C, val_y),
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
    prefetch_factor=4,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True
)

test_loader = DataLoader(
    HierarchicalDataset(test_paths, test_C, test_y),
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
    prefetch_factor=4,
    pin_memory=(device.type == 'cuda'),
    persistent_workers=True
)

model = HierarchicalCNN(num_classes=num_A_classes, num_context_features=1).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
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
            'A_to_idx': A_to_idx,
            'idx_to_A': idx_to_A,
            'num_A_classes': num_A_classes,
            'abundant_classes': abundant_classes,
            'medium_classes': medium_classes,
            'rare_classes': rare_classes
        }, "cnn_A_hierarchical_best.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

checkpoint = torch.load("cnn_A_hierarchical_best.pth", map_location=device)
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
        all_C_vals.extend(context.cpu().numpy().flatten().astype(int).tolist())

all_preds = np.array(all_preds)
all_true = np.array(all_true)
all_C_vals = np.array(all_C_vals)

overall_acc = (all_preds == all_true).mean()
print(f"Overall test accuracy: {overall_acc:.4f}\n")

print("\nClassification report:")
print(classification_report(
    all_true, all_preds,
    target_names=[f"A{idx_to_A[i]}" for i in range(num_A_classes)],
    zero_division=0
))

cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap='Blues', fmt='d')
plt.title("Confusion Matrix (A-level)", fontsize=16)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix_A.png", dpi=200)

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
plt.savefig('training_curves_A.png', dpi=300)

print("\nTraining complete")

