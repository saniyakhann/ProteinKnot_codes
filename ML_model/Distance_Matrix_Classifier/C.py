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

matrices_dir = '/storage/cmstore02/groups/TAPLab/Saniya/Distance_Matrices_256'
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

class DistanceDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        mat = np.load(self.paths[idx])['matrix'].astype(np.float32)
        m = float(mat.mean())
        s = float(mat.std()) + 1e-8
        mat = (mat - m) / s
        x = torch.from_numpy(mat).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

train_paths, temp_paths, y_train, y_temp = train_test_split(paths, labels, test_size=0.3, stratify=labels, random_state=42)
val_paths, test_paths, y_val, y_test = train_test_split(temp_paths, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

batch_size = 8
train_loader = DataLoader(DistanceDataset(train_paths, y_train), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda'))
val_loader = DataLoader(DistanceDataset(val_paths, y_val), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
test_loader = DataLoader(DistanceDataset(test_paths, y_test), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))

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
    model.train()
    train_loss = 0.0
    epoch_start = time.time()

    for batch_idx, (matrices, labels_batch) in enumerate(train_loader):
        matrices = matrices.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(matrices)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx == 0:
            batch_time = time.time() - epoch_start
            est = (batch_time * len(train_loader)) / 60.0
            print(f"First batch: {batch_time:.2f}s | Estimated epoch time: {est:.1f} min")

    avg_train_loss = train_loss / len(train_loader)
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

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_acc = correct / total
    val_accuracies.append(val_acc)

    scheduler.step(avg_val_loss)

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f} | Time: {epoch_time/60:.1f} min")

    if avg_val_loss < best_val_loss - 1e-4:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'cnn_dist_model_best.pth')
        print(f"New model saved (Val Loss: {best_val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience and epoch >= 5:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(range(1, len(train_losses) + 1), train_losses, linewidth=2, label='Training Loss')
ax1.plot(range(1, len(val_losses) + 1), val_losses, linewidth=2, label='Validation Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss Over Time', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(val_accuracies) + 1), val_accuracies, linewidth=2, label='Validation Accuracy')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves_dist_C.png', dpi=300, bbox_inches='tight')
plt.close()

model.load_state_dict(torch.load('cnn_dist_model_best.pth', map_location=device))

print("\nTesting the model")
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for matrices, labels_batch in test_loader:
        matrices = matrices.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)
        outputs = model(matrices)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())
        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

print(f"Test Accuracy: {correct / total:.3f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=[f'Class {idx_to_class[i]}' for i in range(4)], digits=3))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[idx_to_class[i] for i in range(4)], yticklabels=[idx_to_class[i] for i in range(4)])
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix (Distance - C level)')
plt.tight_layout()
plt.savefig('confusion_matrix_dist_C.png', dpi=300, bbox_inches='tight')
plt.close()

torch.save(model.state_dict(), 'cnn_dist_model.pth')
print("\nModel saved to: cnn_dist_model.pth")
