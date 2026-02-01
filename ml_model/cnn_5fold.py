import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

matrices_dir = 'Writhe_Matrices_Selected_500'
csv_path = 'selected_500_per_class.csv'
df = pd.read_csv(csv_path)
df['c_level'] = df['cath_code'].apply(lambda x: int(str(x).split('.')[0]))

class_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
idx_to_class = {0: 1, 1: 2, 2: 3, 3: 4}

paths = []
labels = []
missing = 0
for _, row in df.iterrows():
    fp = os.path.join(matrices_dir, f"{row['domain_id']}.npz")
    if not os.path.exists(fp):
        missing += 1
        continue
    c = int(row['c_level'])
    if c not in class_to_idx:
        continue
    paths.append(fp)
    labels.append(class_to_idx[c])

paths = np.array(paths)
labels = np.array(labels, dtype=np.int64)

print(f"Total proteins with matrices: {len(paths)} (missing: {missing})")
u, counts = np.unique(labels, return_counts=True)
print("Class distribution:")
for k, v in zip(u, counts):
    print(f"  Class {idx_to_class[int(k)]}: {int(v)}")

def downsample_block_mean(mat, out=256):
    H, W = mat.shape
    newH = int(np.ceil(H / out) * out)
    newW = int(np.ceil(W / out) * out)
    padH = newH - H
    padW = newW - W
    if padH or padW:
        mat = np.pad(mat, ((0, padH), (0, padW)), mode="constant", constant_values=0.0)
    mat = mat.reshape(out, newH // out, out, newW // out).mean(axis=(1, 3))
    return mat

class WritheDataset(Dataset):
    def __init__(self, paths, labels, target_size=256):
        self.paths = paths
        self.labels = labels
        self.target_size = target_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = np.load(self.paths[idx])['matrix'].astype(np.float32)
        mat = downsample_block_mean(mat, self.target_size)
        m = float(mat.mean())
        s = float(mat.std()) + 1e-8
        mat = (mat - m) / s
        x = torch.from_numpy(mat).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

class CNN(nn.Module):
    def __init__(self, num_classes=4):
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
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def eval_loader(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for matrices, labels_batch in loader:
            matrices = matrices.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            outputs = model(matrices)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            total += labels_batch.size(0)
            correct += (preds == labels_batch).sum().item()
    acc = correct / max(1, total)
    return acc, np.array(all_labels), np.array(all_preds)

target_size = 256
batch_size = 8
max_epochs = 200
patience = 10

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_rows = []
cm_sum = np.zeros((4, 4), dtype=np.int64)
accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels), start=1):

    print(f"Fold {fold}/5")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")


    train_paths = paths[train_idx]
    train_labels = labels[train_idx]
    val_paths = paths[val_idx]
    val_labels = labels[val_idx]

    train_loader = DataLoader(
        WritheDataset(train_paths, train_labels, target_size=target_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        WritheDataset(val_paths, val_labels, target_size=target_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    model = CNN(num_classes=4).to(device)

    class_counts = np.bincount(train_labels, minlength=4).astype(np.float32)
    class_weights = (class_counts.sum() / (class_counts + 1e-8))
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    patience_counter = 0
    best_path = f"cnn_model_best_fold{fold}.pth"

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        epoch_start = time.time()

        for matrices, labels_batch in train_loader:
            matrices = matrices.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(matrices)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for matrices, labels_batch in val_loader:
                matrices = matrices.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                outputs = model(matrices)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= 5:
                break

        if (epoch + 1) % 5 == 0:
            mins = (time.time() - epoch_start) / 60.0
            print(f"Fold {fold} | Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | {mins:.2f} min")

        gc.collect()

    model.load_state_dict(torch.load(best_path, map_location=device))
    val_acc, y_true, y_pred = eval_loader(model, val_loader)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_sum += cm
    accs.append(val_acc)

    fold_rows.append({
        "fold": fold,
        "val_accuracy": val_acc,
        "best_val_loss": best_val_loss,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
    })

    print(f"\nFold {fold} Validation Accuracy: {val_acc:.4f}")
    print(f"Fold {fold} Best Val Loss: {best_val_loss:.4f}")

results = pd.DataFrame(fold_rows)
mean_acc = float(np.mean(accs))
std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
results.loc[len(results)] = {"fold": "mean", "val_accuracy": mean_acc, "best_val_loss": np.nan, "train_size": np.nan, "val_size": np.nan}
results.loc[len(results)] = {"fold": "std", "val_accuracy": std_acc, "best_val_loss": np.nan, "train_size": np.nan, "val_size": np.nan}
results.to_csv("cv_results_5fold.csv", index=False)


print("5-Fold CV Summary")

print(results)
print(f"\nMean CV Accuracy: {mean_acc:.4f}")
print(f"Std CV Accuracy:  {std_acc:.4f}")

np.save("cv_confusion_matrix_sum.npy", cm_sum)

cm_norm = cm_sum / cm_sum.sum(axis=1, keepdims=True)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=[1, 2, 3, 4],
    yticklabels=[1, 2, 3, 4],
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Mean Confusion Matrix (5-Fold CV, Row-Normalized)")
plt.tight_layout()
plt.savefig("cv_confusion_matrix_norm.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 4))
plt.bar([str(i) for i in range(1, 6)], accs)
plt.ylim(0.0, 1.0)
plt.xlabel("Fold")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy per Fold (5-Fold CV)")
plt.tight_layout()
plt.savefig("cv_accuracy_bar.png", dpi=300, bbox_inches="tight")
plt.close()
