import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writhe_dir = 'Writhe_Matrices_256'
csv_path = 'H_level_domain_ids_writhe.csv'

df = pd.read_csv(csv_path)
proteins_df = pd.read_csv('proteins_final.csv')


def parse_cath(code):
    parts = str(code).strip().split('.')
    if len(parts) < 4:
        return None, None, None, None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except:
        return None, None, None, None


parsed = proteins_df['cath_code'].apply(parse_cath)
proteins_df['C'] = parsed.apply(lambda x: x[0])
proteins_df['A'] = parsed.apply(lambda x: x[1])
proteins_df['T'] = parsed.apply(lambda x: x[2])
proteins_df['H'] = parsed.apply(lambda x: x[3])
proteins_df = proteins_df.dropna(subset=['C', 'A', 'T', 'H']).copy()
proteins_df['C'] = proteins_df['C'].astype(int)
proteins_df['A'] = proteins_df['A'].astype(int)
proteins_df['T'] = proteins_df['T'].astype(int)
proteins_df['H'] = proteins_df['H'].astype(int)

df = df.merge(proteins_df[['domain_id', 'C', 'A', 'T', 'H']], on='domain_id', how='left')
df = df.dropna(subset=['C', 'A', 'T', 'H']).copy()


def downsample_block_mean(mat, out=256):
    h, w = mat.shape
    new_h = int(np.ceil(h / out) * out)
    new_w = int(np.ceil(w / out) * out)
    pad_h, pad_w = new_h - h, new_w - w
    if pad_h or pad_w:
        mat = np.pad(mat, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0.0)
    return mat.reshape(out, new_h // out, out, new_w // out).mean(axis=(1, 3))


class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes, num_context_features=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=False), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.context_mlp = nn.Sequential(nn.Linear(num_context_features, 32), nn.ReLU(), nn.Linear(32, 64))
        self.classifier = nn.Sequential(nn.Linear(256 + 64, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x, context):
        img_feat = torch.flatten(self.pool(self.features(x)), 1)
        return self.classifier(torch.cat([img_feat, self.context_mlp(context)], dim=1))


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = self.activations = None
        model.features[-1].register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))
        model.features[-1].register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def compute(self, x, context, class_idx):
        self.model.zero_grad()
        output = self.model(x, context)
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


checkpoint = torch.load('/home/s2147128/cnn_H_hierarchical_best.pth', map_location=device, weights_only=False)
h_to_idx = {v: k for k, v in checkpoint['idx_to_H'].items()}
idx_to_h = checkpoint['idx_to_H']

model = HierarchicalCNN(num_classes=checkpoint['num_H_classes']).to(device)
model.load_state_dict(checkpoint['model_state'])
model.eval()
gc = GradCAM(model)

results = []
for i, row in df.iterrows():
    fp = os.path.join(writhe_dir, f"{row['domain_id']}.npz")
    if not os.path.exists(fp):
        continue

    true_h = int(row['H'])
    true_idx = h_to_idx.get(true_h, -1)
    if true_idx == -1:
        continue

    mat_raw = np.load(fp)['matrix'].astype(np.float32)
    mat_raw = mat_raw[:-1, :-1]
    mat_256 = downsample_block_mean(mat_raw, 256)
    m, s = float(mat_256.mean()), float(mat_256.std()) + 1e-8
    mat_norm = (mat_256 - m) / s
    x = torch.from_numpy(mat_norm).unsqueeze(0).unsqueeze(0).to(device)

    c_norm = (int(row['C']) - 1) / 3.0
    a_norm = int(row['A']) / 170.0
    t_norm = int(row['T']) / 4090.0
    context = torch.tensor([[c_norm, a_norm, t_norm]], dtype=torch.float32).to(device)

    cam, pred, conf = gc.compute(x, context, true_idx)
    cam_256 = cv2.resize(cam, (256, 256))
    dar = compute_dar(cam_256)

    results.append({
        'domain_id': row['domain_id'],
        'C_val': int(row['C']),
        'true_H': true_h,
        'pred_H': idx_to_h.get(pred, -1),
        'confidence': round(conf, 4),
        'correct': int(pred == true_idx),
        'DAR': round(dar, 4)
    })

    if i % 500 == 0:
        print(f'{i}/{len(df)}')

res_df = pd.DataFrame(results)
res_df.to_csv('gradcam_results_writhe_H.csv', index=False)
print('saved: gradcam_results_writhe_H.csv')
print(f'overall accuracy: {res_df["correct"].mean():.3f}')
print('\nmedian dar by cath class:')
print(res_df.groupby('C_val')['DAR'].median().round(3))
print('\ncorrect vs wrong dar:')
print(res_df.groupby('correct')['DAR'].median().round(3))
