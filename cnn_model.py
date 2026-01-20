import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
import gc  
import matplotlib.pyplot as plt
import seaborn as sns

#load the data
matrices_dir = 'Writhe_Matrices_Padded'
csv_path = 'single_domain_proteins_complete.csv'

df = pd.read_csv(csv_path)
df['c_level'] = df['cath_code'].apply(lambda x: int(str(x).split('.')[0]))

matrix_files = [f for f in os.listdir(matrices_dir) if f.endswith('.npz')]
matrix_ids = [f.replace('.npz', '') for f in matrix_files]

df_matched = df[df['domain_id'].isin(matrix_ids)]
print(f"Total available proteins: {len(df_matched)}")
df_matched = df_matched.groupby('c_level', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 1000), random_state=42)
)

if len(df_matched) > 1000:
    df_matched = df_matched.sample(n=1000, random_state=42)

#load matrices and labels
X = []
y = []

class_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 6: 4}
idx_to_class = {0: 1, 1: 2, 2: 3, 3: 4, 4: 6}

for idx, (_, row) in enumerate(df_matched.iterrows()):
    filepath = os.path.join(matrices_dir, row['domain_id'] + '.npz')
    matrix = np.load(filepath)['matrix']
    X.append(matrix)
    y.append(class_to_idx[row['c_level']])
    
    if (idx + 1) % 1000 == 0:
        print(f"  Loaded {idx + 1}/{len(df_matched)}")

X = np.array(X)
y = np.array(y)

print(f"Loaded: {len(X)} samples")

#calculate class weights for imbalanced data
class_counts = np.bincount(y)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights = torch.FloatTensor(class_weights)

print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights}")

#train/val/test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

#dataset and dataloader
class WritheDataset(Dataset):
    def __init__(self, matrices, labels):
        self.matrices = torch.FloatTensor(matrices).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.matrices[idx], self.labels[idx]

train_loader = DataLoader(WritheDataset(X_train, y_train), batch_size=4, shuffle=True, num_workers=0, pin_memory=False) 
val_loader = DataLoader(WritheDataset(X_val, y_val), batch_size=4, num_workers=0, pin_memory=False)  
test_loader = DataLoader(WritheDataset(X_test, y_test), batch_size=4, num_workers=0, pin_memory=False) 

#cnn model
class CNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CNN(num_classes=5)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

#training
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining the model")
for epoch in range(20):
    #train
    model.train()
    train_loss = 0
    for matrices, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(matrices)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    gc.collect()  
    
    #validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for matrices, labels in val_loader:
            outputs = model(matrices)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    print(f"Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.3f}")

#testing the model
print("\nTesting the model")
model.eval()
correct = 0
total = 0
all_preds = [] 
all_labels = []  
with torch.no_grad():
    for matrices, labels in test_loader:
        outputs = model(matrices)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())  
        all_labels.extend(labels.cpu().numpy())  
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct/total:.3f}")

#plotting confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[idx_to_class[i] for i in range(5)],
            yticklabels=[idx_to_class[i] for i in range(5)])
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

#per class accuracy
print("\nPer-class accuracy:")
for i in range(5):
    if cm[i].sum() > 0:
        class_acc = cm[i][i] / cm[i].sum()
        print(f"Class {idx_to_class[i]}: {class_acc:.3f} ({cm[i][i]}/{cm[i].sum()})")

#save model 
torch.save(model.state_dict(), 'cnn_model.pth')
print("\nModel saved to: cnn_model.pth")
