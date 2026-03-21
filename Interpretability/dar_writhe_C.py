import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writhe_dir = 'Writhe_Matrices_Selected_500'
writhe_model_path = 'cnn_model_best.pth'
csv_path = 'selected_500_per_class.csv'

class_names = {0: 'Mainly Alpha', 1: 'Mainly Beta', 2: 'Alpha/Beta', 3: 'Few SS'}
class_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}

df = pd.read_csv(csv_path)
df['c_level'] = df['cath_code'].apply(lambda x: int(str(x).split('.')[0]))

paths, labels, domain_ids = [], [], []
for _, row in df.iterrows():
    fp = os.path.join(writhe_dir, f"{row['domain_id']}.npz")
    if not os.path.exists(fp):
        continue
    c = int(row['c_level'])
    if c not in class_to_idx:
        continue
    paths.append(fp)
    labels.append(class_to_idx[c])
    domain_ids.append(row['domain_id'])


def downsample_block_mean(mat, out=256):
    h, w = mat.shape
    new_h = int(np.ceil(h / out) * out)
    new_w = int(np.ceil(w / out) * out)
    pad_h, pad_w = new_h - h, new_w - w
    if pad_h or pad_w:
        mat = np.pad(mat, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0.0)
    return mat.reshape(out, new_h // out, out, new_w // out).mean(axis=(1, 3))


class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(torch.flatten(self.pool(self.features(x)), 1)))


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = self.activations = None
        model.features[-1].register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        model.features[-1].register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def compute(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        output[0, class_idx].backward()
        weights = self.gradients[0].mean(dim=(1, 2))
        cam = sum(w * a for w, a in zip(weights, self.activations[0]))
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        pred = output.argmax(dim=1).item()
        conf = output.softmax(dim=1)[0, pred].item()
        return cam.cpu().numpy(), pred, conf


def compute_dar(cam, band_fraction=0.1):
    h, w = cam.shape
    band = max(1, int(h * band_fraction))
    total = cam.sum()
    if total == 0:
        return 0.0
    return float(sum(
        cam[i, max(0, i - band):min(w, i + band + 1)].sum()
        for i in range(h)
    ) / total)


model = CNN(num_classes=4).to(device)
model.load_state_dict(torch.load(writhe_model_path, map_location=device, weights_only=False))
model.eval()
gc = GradCAM(model)

results = []
for i, (fp, lbl, did) in enumerate(zip(paths, labels, domain_ids)):
    mat_raw = np.load(fp)['matrix'].astype(np.float32)
    mat_raw = mat_raw[:-1, :-1]
    true_size = mat_raw.shape[0]
    mat_256 = downsample_block_mean(mat_raw, 256)
    m, s = float(mat_256.mean()), float(mat_256.std()) + 1e-8
    mat_norm = (mat_256 - m) / s
    x = torch.from_numpy(mat_norm).unsqueeze(0).unsqueeze(0).to(device)

    cam, pred, conf = gc.compute(x, lbl)
    cam_256 = cv2.resize(cam, (256, 256))
    dar = compute_dar(cam_256)

    results.append({
        'domain_id': did,
        'true_class': lbl,
        'true_class_name': class_names[lbl],
        'pred_class': pred,
        'pred_class_name': class_names[pred],
        'confidence': round(conf, 4),
        'correct': int(pred == lbl),
        'DAR': round(dar, 4),
        'protein_size': true_size
    })

    if i % 100 == 0:
        print(f'{i}/{len(paths)}')

pd.DataFrame(results).to_csv('gradcam_results_writhe_C.csv', index=False)
print('saved: gradcam_results_writhe_C.csv')
