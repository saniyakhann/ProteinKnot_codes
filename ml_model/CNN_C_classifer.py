#most recent C classifying model, accuracy is coming out as 70%
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from sklearn.metrics import confusion_matrix, classification_report
import gc  
import matplotlib.pyplot as plt
import seaborn as sns
import time

#device configuration 
device = torch.device('cpu')
print(f"Using device: {device}\n")

#load the data
matrices_dir = 'Writhe_Matrices_Selected_200'
csv_path = '200_selected_proteins_C.csv'

df_matched = pd.read_csv(csv_path)
df_matched['c_level'] = df_matched['cath_code'].apply(lambda x: int(str(x).split('.')[0]))

print(f"Total proteins for training: {len(df_matched)}")
print("Class distribution:")
print(df_matched['c_level'].value_counts().sort_index())

#load matrices and labels
X = []
y = []

class_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
idx_to_class = {0: 1, 1: 2, 2: 3, 3: 4}

for idx, (_, row) in enumerate(df_matched.iterrows()):
    filepath = os.path.join(matrices_dir, row['domain_id'] + '.npz')
    matrix = np.load(filepath)['matrix']
    
#downsampling 
    target_size = 1000
    downsample_factor = target_size / matrix.shape[0]
    matrix = zoom(matrix, downsample_factor, order=3)
    
    #convert to float32
    matrix = matrix.astype(np.float32) #rounding to 2 decimal places
    
    X.append(matrix)
    y.append(class_to_idx[row['c_level']])
    
    if (idx + 1) % 100 == 0:
        print(f"  Loaded {idx + 1}/{len(df_matched)}")
    
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
        self.matrices = matrices
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        matrix = torch.FloatTensor(self.matrices[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return matrix, label

train_loader = DataLoader(WritheDataset(X_train, y_train), batch_size=10, shuffle=True, num_workers=0, pin_memory=False) 
val_loader = DataLoader(WritheDataset(X_val, y_val), batch_size=10, num_workers=0, pin_memory=False)  
test_loader = DataLoader(WritheDataset(X_test, y_test), batch_size=10, num_workers=0, pin_memory=False) 

#cnn model
class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256) #BatchNorm reduces internal covariate shift and improves training stability
        
        self.pool = nn.MaxPool2d(2, 2) #maxpool reduces spatial size and keep the strongest signal
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  #to control feature map size before fc layers to reduce no. of parameters
        
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))  
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))  
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))  
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))  
        x = self.adaptive_pool(x) 
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CNN(num_classes=4)
model = model.to(device)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

#training
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

#early stopping parameters
best_val_acc = 0
patience = 8
patience_counter = 0

#tracking metrics
train_losses = []
val_accuracies = []

print("\nTraining the model")
for epoch in range(1000):  #increased max epochs since we have early stopping
    #train
    model.train()
    train_loss = 0
    epoch_start = time.time()
    
    for batch_idx, (matrices, labels) in enumerate(train_loader):
        batch_start = time.time()
        
        matrices = matrices.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(matrices)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if batch_idx == 0:
            batch_time = time.time() - batch_start
       
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Completed {batch_idx+1}/{len(train_loader)} batches")
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch training time: {epoch_time/60:.1f} minutes")
    
    gc.collect()  
    
    #validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for matrices, labels in val_loader:
            matrices = matrices.to(device)
            labels = labels.to(device)
            outputs = model(matrices)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    val_accuracies.append(val_acc)
    
    scheduler.step(val_acc)
    
    print(f"Epoch {epoch+1:2d} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.3f}")
    
    #early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        #save best model
        torch.save(model.state_dict(), 'cnn_model_best.pth')
        print(f" New model saved (Val Acc: {best_val_acc:.3f})")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience and epoch >= 5:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.3f}")
            break


#plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

#loss curve
ax1.plot(range(1, len(train_losses)+1), train_losses, 'b-', linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

#accuracy curve
ax2.plot(range(1, len(val_accuracies)+1), val_accuracies, 'g-', linewidth=2, label='Validation Accuracy')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.close()

#load best model for testing
model.load_state_dict(torch.load('cnn_model_best.pth'))

#testing the model
print("\nTesting the model")
model.eval()
correct = 0
total = 0
all_preds = [] 
all_labels = []  
with torch.no_grad():
    for matrices, labels in test_loader:
        matrices = matrices.to(device)
        labels = labels.to(device)
        outputs = model(matrices)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())  
        all_labels.extend(labels.cpu().numpy())  
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct/total:.3f}")

#classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, 
                           target_names=[f'Class {idx_to_class[i]}' for i in range(4)],
                           digits=3))

#plotting confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[idx_to_class[i] for i in range(4)],
            yticklabels=[idx_to_class[i] for i in range(4)])
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

#save final model 
torch.save(model.state_dict(), 'cnn_model.pth')
print("\nModel saved to: cnn_model.pth")
