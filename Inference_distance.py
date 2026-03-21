import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

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

class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes, num_context_features):
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

    def forward(self, matrix, context):
        img_feat = self.features(matrix)
        img_feat = self.pool(img_feat)
        img_feat = torch.flatten(img_feat, 1)
        ctx_feat = self.context_mlp(context)
        combined = torch.cat([img_feat, ctx_feat], dim=1)
        return self.classifier(combined)

class InferenceDataset(Dataset):
    def __init__(self, paths, C_vals, A_vals, T_vals, H_vals):
        self.paths = paths
        self.C_vals = C_vals
        self.A_vals = A_vals
        self.T_vals = T_vals
        self.H_vals = H_vals

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        mat = np.load(self.paths[idx])['matrix'].astype(np.float32)
        m = float(mat.mean())
        s = float(mat.std()) + 1e-8
        mat = (mat - m) / s
        x = torch.from_numpy(mat).unsqueeze(0)
        return x, self.C_vals[idx], self.A_vals[idx], self.T_vals[idx], self.H_vals[idx]

C_model = CNN(num_classes=4).to(device)
C_model.load_state_dict(torch.load('cnn_dist_model_best.pth', map_location=device, weights_only=False))
C_model.eval()

A_checkpoint = torch.load('cnn_dist_A_best.pth', map_location=device, weights_only=False)
A_model = HierarchicalCNN(num_classes=A_checkpoint['num_A_classes'], num_context_features=1).to(device)
A_model.load_state_dict(A_checkpoint['model_state'])
A_model.eval()
A_to_idx = A_checkpoint['A_to_idx']
idx_to_A = A_checkpoint['idx_to_A']

T_checkpoint = torch.load('cnn_dist_T_best.pth', map_location=device, weights_only=False)
T_model = HierarchicalCNN(num_classes=T_checkpoint['num_T_classes'], num_context_features=2).to(device)
T_model.load_state_dict(T_checkpoint['model_state'])
T_model.eval()
T_to_idx = T_checkpoint['T_to_idx']
idx_to_T = T_checkpoint['idx_to_T']

H_checkpoint = torch.load('cnn_dist_H_best.pth', map_location=device, weights_only=False)
H_model = HierarchicalCNN(num_classes=H_checkpoint['num_H_classes'], num_context_features=3).to(device)
H_model.load_state_dict(H_checkpoint['model_state'])
H_model.eval()
idx_to_H = H_checkpoint['idx_to_H']

matrices_dir = '/storage/cmstore02/groups/TAPLab/Saniya/Distance_Matrices_256'
csv_path = '/home/s2147128/proteins_final.csv'
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

H_to_new_idx = {v: k for k, v in idx_to_H.items()}

paths = []
C_vals = []
A_vals = []
T_vals = []
H_vals = []

for _, row in df.iterrows():
    fp = os.path.join(matrices_dir, f"{row['domain_id']}.npz")
    if not os.path.exists(fp):
        continue
    if row['T'] not in T_to_idx or row['H'] not in H_to_new_idx:
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

C_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
idx_to_C = {0: 1, 1: 2, 2: 3, 3: 4}

C_labels = np.array([C_to_idx[c] for c in C_vals])
A_labels = np.array([A_to_idx[a] if a in A_to_idx else -1 for a in A_vals])
T_labels = np.array([T_to_idx[t] if t in T_to_idx else -1 for t in T_vals])
H_labels = np.array([H_to_new_idx[h] if h in H_to_new_idx else -1 for h in H_vals])

valid_mask = (A_labels != -1) & (T_labels != -1) & (H_labels != -1)
paths = paths[valid_mask]
C_labels = C_labels[valid_mask]
A_labels = A_labels[valid_mask]
T_labels = T_labels[valid_mask]
H_labels = H_labels[valid_mask]

np.random.seed(42)
test_size = min(5000, len(paths))
test_indices = np.random.choice(len(paths), test_size, replace=False)

test_paths = paths[test_indices]
test_C = C_labels[test_indices]
test_A = A_labels[test_indices]
test_T = T_labels[test_indices]
test_H = H_labels[test_indices]

print(f"Test set: {len(test_paths)} proteins\n")

test_loader = DataLoader(
    InferenceDataset(test_paths, test_C, test_A, test_T, test_H),
    batch_size=32, shuffle=False, num_workers=0
)

C_preds = []
A_preds = []
T_preds = []
H_preds = []
C_true = []
A_true = []
T_true = []
H_true = []

with torch.no_grad():
    for matrices, true_C_batch, true_A_batch, true_T_batch, true_H_batch in test_loader:
        matrices = matrices.to(device)

        C_output = C_model(matrices)
        C_pred_batch = C_output.argmax(dim=1)

        C_pred_values = torch.tensor([idx_to_C[idx.item()] for idx in C_pred_batch], device=device)
        C_pred_norm = (C_pred_values - 1.0) / 3.0
        pred_C_context = C_pred_norm.unsqueeze(1).float()
        A_output = A_model(matrices, pred_C_context)
        A_pred_batch = A_output.argmax(dim=1)

        A_pred_values = torch.tensor([idx_to_A[idx.item()] for idx in A_pred_batch], device=device)
        A_pred_norm = A_pred_values / 170.0
        pred_CA_context = torch.stack([C_pred_norm, A_pred_norm], dim=1).float()
        T_output = T_model(matrices, pred_CA_context)
        T_pred_batch = T_output.argmax(dim=1)

        T_pred_values = torch.tensor([idx_to_T[idx.item()] for idx in T_pred_batch], device=device)
        T_pred_norm = T_pred_values / 4090.0
        pred_CAT_context = torch.stack([C_pred_norm, A_pred_norm, T_pred_norm], dim=1).float()
        H_output = H_model(matrices, pred_CAT_context)
        H_pred_batch = H_output.argmax(dim=1)

        C_preds.extend(C_pred_batch.cpu().numpy())
        A_preds.extend(A_pred_batch.cpu().numpy())
        T_preds.extend(T_pred_batch.cpu().numpy())
        H_preds.extend(H_pred_batch.cpu().numpy())
        C_true.extend(true_C_batch.numpy())
        A_true.extend(true_A_batch.numpy())
        T_true.extend(true_T_batch.numpy())
        H_true.extend(true_H_batch.numpy())

C_preds = np.array(C_preds)
A_preds = np.array(A_preds)
T_preds = np.array(T_preds)
H_preds = np.array(H_preds)
C_true = np.array(C_true)
A_true = np.array(A_true)
T_true = np.array(T_true)
H_true = np.array(H_true)

C_acc = accuracy_score(C_true, C_preds)
A_acc = accuracy_score(A_true, A_preds)
T_acc = accuracy_score(T_true, T_preds)
H_acc = accuracy_score(H_true, H_preds)

CA_acc = ((C_preds == C_true) & (A_preds == A_true)).sum() / len(C_true)
CAT_acc = ((C_preds == C_true) & (A_preds == A_true) & (T_preds == T_true)).sum() / len(C_true)
CATH_acc = ((C_preds == C_true) & (A_preds == A_true) & (T_preds == T_true) & (H_preds == H_true)).sum() / len(C_true)

print("Results:")
print(f"  C accuracy: {C_acc:.4f}")
print(f"  A accuracy: {A_acc:.4f}")
print(f"  T accuracy: {T_acc:.4f}")
print(f"  H accuracy: {H_acc:.4f}")
print(f"  C+A both correct: {CA_acc:.4f}")
print(f"  C+A+T all correct: {CAT_acc:.4f}")
print(f"  C+A+T+H all correct: {CATH_acc:.4f}")

print("\nC-level breakdown:")
for c_idx in range(4):
    mask = (C_true == c_idx)
    if mask.sum() == 0:
        continue
    c_val = idx_to_C[c_idx]
    print(f"  C={c_val}: A={accuracy_score(A_true[mask], A_preds[mask]):.4f}, T={accuracy_score(T_true[mask], T_preds[mask]):.4f}, H={accuracy_score(H_true[mask], H_preds[mask]):.4f} (n={mask.sum()})")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

cm_C = confusion_matrix(C_true, C_preds)
sns.heatmap(cm_C, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=[idx_to_C[i] for i in range(4)],
            yticklabels=[idx_to_C[i] for i in range(4)])
axes[0, 0].set_title(f'C-level (Acc: {C_acc:.3f})', fontsize=14)
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('True')

cm_A = confusion_matrix(A_true, A_preds)
cm_A_norm = cm_A.astype('float') / (cm_A.sum(axis=1, keepdims=True) + 1e-10)
sns.heatmap(cm_A_norm, annot=False, cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title(f'A-level (Acc: {A_acc:.3f})', fontsize=14)
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('True')

cm_T = confusion_matrix(T_true, T_preds)
cm_T_norm = cm_T.astype('float') / (cm_T.sum(axis=1, keepdims=True) + 1e-10)
sns.heatmap(cm_T_norm, annot=False, cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title(f'T-level (Acc: {T_acc:.3f})', fontsize=14)
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('True')

cm_H = confusion_matrix(H_true, H_preds)
cm_H_norm = cm_H.astype('float') / (cm_H.sum(axis=1, keepdims=True) + 1e-10)
sns.heatmap(cm_H_norm, annot=False, cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title(f'H-level (Acc: {H_acc:.3f})', fontsize=14)
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('True')

plt.tight_layout()
plt.savefig('inference_CATH_dist.png', dpi=300)
print("\nSaved: inference_CATH_dist.png")

pd.DataFrame([{
    'C_accuracy': C_acc,
    'A_accuracy': A_acc,
    'T_accuracy': T_acc,
    'H_accuracy': H_acc,
    'CA_both_correct': CA_acc,
    'CAT_all_correct': CAT_acc,
    'CATH_all_correct': CATH_acc
}]).to_csv('inference_CATH_dist_results.csv', index=False)
print("Saved: inference_CATH_dist_results.csv")