import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import time

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

target_size = 256
train_paths, temp_paths, y_train, y_temp = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=42)
val_paths, test_paths, y_val, y_test = train_test_split(temp_paths, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

batch_size = 8
train_loader = DataLoader(WritheDataset(train_paths, y_train, target_size=target_size), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
val_loader = DataLoader(WritheDataset(val_paths, y_val, target_size=target_size), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
test_loader = DataLoader(WritheDataset(test_paths, y_test, target_size=target_size), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

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

model = CNN(num_classes=4).to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

class_counts = np.bincount(y_train, minlength=4).astype(np.float32)
class_weights = (class_counts.sum() / (class_counts + 1e-8))
class_weights = class_weights / class_weights.mean()
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

# criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

best_val_loss = float('inf')
patience = 10
patience_counter = 0

train_losses = []
val_losses = []
val_accuracies = []

max_epochs = 200

print("\nTraining the model")
for epoch in range(max_epochs):
    print(f"\n=== Epoch {epoch+1} started ===")
    model.train()
    train_loss = 0.0
    epoch_start = time.time()

    for batch_idx, (matrices, labels_batch) in enumerate(train_loader):
        matrices = matrices.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(matrices)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx == 0:
            batch_time = time.time() - epoch_start
            est = (batch_time * len(train_loader)) / 60.0
            print(f"First batch: {batch_time:.2f}s | Estimated epoch time: {est:.1f} min")

    avg_train_loss = train_loss / max(1, len(train_loader))
    train_losses.append(avg_train_loss)

    gc.collect()

    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for matrices, labels_batch in val_loader:
            matrices = matrices.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            outputs = model(matrices)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    avg_val_loss = val_loss / max(1, len(val_loader))
