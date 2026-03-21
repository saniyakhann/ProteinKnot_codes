import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

domain_id = '5x9vD01'
true_class = 0
crop = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writhe_dir = 'Writhe_Matrices_Selected_500'
dist_dir = '/storage/cmstore02/groups/TAPLab/Saniya/Distance_Matrices_256'
writhe_model_path = 'cnn_model_best.pth'
dist_model_path = '/home/s2147128/cnn_dist_model_best.pth'

class_names = {0: 'Mainly Alpha', 1: 'Mainly Beta', 2: 'Alpha/Beta', 3: 'Few SS'}


def downsample_block_mean(mat, out=256):
    h, w = mat.shape
    new_h = int(np.ceil(h / out) * out)
    new_w = int(np.ceil(w / out) * out)
    pad_h, pad_w = new_h - h, new_w - w
    if pad_h or pad_w:
        mat = np.pad(mat, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0.0)
    return mat.reshape(out, new_h // out, out, new_w // out).mean(axis=(1, 3))


def load_writhe_for_model(path):
    mat = np.load(path)['matrix'].astype(np.float32)
    mat = mat[:-1, :-1]
    mat = downsample_block_mean(mat, 256)
    m, s = float(mat.mean()), float(mat.std()) + 1e-8
    return (mat - m) / s


def load_dist_for_model(path):
    mat = np.load(path)['matrix'].astype(np.float32)
    m, s = float(mat.mean()), float(mat.std()) + 1e-8
    return (mat - m) / s


def load_writhe_true(path):
    mat = np.load(path)['matrix'].astype(np.float32)
    mat = mat[:-1, :-1]
    nonzero_rows = np.any(mat != 0, axis=1)
    nonzero_cols = np.any(mat != 0, axis=0)
    row_end = int(np.where(nonzero_rows)[0][-1]) + 1 if nonzero_rows.any() else mat.shape[0]
    col_end = int(np.where(nonzero_cols)[0][-1]) + 1 if nonzero_cols.any() else mat.shape[1]
    mat = mat[:row_end, :col_end]
    m, s = float(mat.mean()), float(mat.std()) + 1e-8
    return (mat - m) / s


def load_dist_true(path):
    mat = np.load(path)['matrix'].astype(np.float32)
    nonzero_rows = np.any(mat != 0, axis=1)
    nonzero_cols = np.any(mat != 0, axis=0)
    row_end = int(np.where(nonzero_rows)[0][-1]) + 1 if nonzero_rows.any() else mat.shape[0]
    col_end = int(np.where(nonzero_cols)[0][-1]) + 1 if nonzero_cols.any() else mat.shape[1]
    mat = mat[:row_end, :col_end]
    m, s = float(mat.mean()), float(mat.std()) + 1e-8
    return (mat - m) / s


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
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


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
    diag = sum(
        cam[i, max(0, i - band):min(w, i + band + 1)].sum()
        for i in range(h)
    )
    return float(diag / total)


def apply_crop(mat, cam, crop):
    if crop is None:
        return mat, cam
    r = min(crop, mat.shape[0])
    c = min(crop, mat.shape[1])
    return mat[:r, :c], cv2.resize(cam[:r, :c], (c, r))


w_model = CNN().to(device)
w_model.load_state_dict(torch.load(writhe_model_path, map_location=device, weights_only=False))
w_model.eval()
w_gc = GradCAM(w_model)

d_model = CNN().to(device)
d_model.load_state_dict(torch.load(dist_model_path, map_location=device, weights_only=False))
d_model.eval()
d_gc = GradCAM(d_model)

w_true = load_writhe_true(f'{writhe_dir}/{domain_id}.npz')
d_true = load_dist_true(f'{dist_dir}/{domain_id}.npz')

w_256 = load_writhe_for_model(f'{writhe_dir}/{domain_id}.npz')
d_256 = load_dist_for_model(f'{dist_dir}/{domain_id}.npz')

print(f'{domain_id} | true: {class_names[true_class]}')
print(f'writhe input: {w_true.shape[0]}x{w_true.shape[1]} (display), {w_256.shape[0]}x{w_256.shape[1]} (model)')
print(f'distance input: {d_true.shape[0]}x{d_true.shape[1]} (display), {d_256.shape[0]}x{d_256.shape[1]} (model)')

xw = torch.from_numpy(w_256).unsqueeze(0).unsqueeze(0).to(device)
w_cam_256, w_pred, w_conf = w_gc.compute(xw, true_class)

xd = torch.from_numpy(d_256).unsqueeze(0).unsqueeze(0).to(device)
d_cam_256, d_pred, d_conf = d_gc.compute(xd, true_class)

w_cam_true = cv2.resize(w_cam_256, (w_true.shape[1], w_true.shape[0]))
d_cam_true = cv2.resize(d_cam_256, (d_true.shape[1], d_true.shape[0]))

w_dar = compute_dar(w_cam_true)
d_dar = compute_dar(d_cam_true)

print(f'writhe  -> {class_names[w_pred]} (conf {w_conf:.3f}, DAR {w_dar:.3f})')
print(f'distance -> {class_names[d_pred]} (conf {d_conf:.3f}, DAR {d_dar:.3f})')

w_mat_plot, w_cam_plot = apply_crop(w_true, w_cam_true, crop)
d_mat_plot, d_cam_plot = apply_crop(d_true, d_cam_true, crop)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    f'{domain_id} — True: {class_names[true_class]}\n'
    f'Writhe pred: {class_names[w_pred]} (conf: {w_conf:.2f}, DAR: {w_dar:.3f})   '
    f'Distance pred: {class_names[d_pred]} (conf: {d_conf:.2f}, DAR: {d_dar:.3f})',
    fontsize=11, fontweight='bold'
)

axes[0, 0].imshow(w_mat_plot, cmap='RdBu_r', aspect='auto')
axes[0, 0].set_title('Writhe Matrix (true size)', fontsize=10)
axes[0, 0].set_xlabel('Segment j')
axes[0, 0].set_ylabel('Segment i')

axes[0, 1].imshow(w_cam_plot, cmap='hot', aspect='auto')
axes[0, 1].set_title('Writhe Grad-CAM', fontsize=10)
axes[0, 1].set_xlabel('Segment j')

axes[1, 0].imshow(d_mat_plot, cmap='RdBu_r', aspect='auto')
axes[1, 0].set_title('Distance Matrix (true size)', fontsize=10)
axes[1, 0].set_xlabel('Residue j')
axes[1, 0].set_ylabel('Residue i')

axes[1, 1].imshow(d_cam_plot, cmap='hot', aspect='auto')
axes[1, 1].set_title('Distance Grad-CAM', fontsize=10)
axes[1, 1].set_xlabel('Residue j')

plt.tight_layout()
crop_str = f'_crop{crop}' if crop else ''
outname = f'compare_{domain_id}_class{true_class}{crop_str}.png'
plt.savefig(outname, dpi=200, bbox_inches='tight')
plt.close()
print(f'saved: {outname}')
