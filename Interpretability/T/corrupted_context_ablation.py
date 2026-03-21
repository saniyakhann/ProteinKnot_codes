import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_cath(code):
    parts = str(code).strip().split('.')
    if len(parts) < 3:
        return None, None, None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except:
        return None, None, None


def load_writhe(path):
    mat = np.load(path)['matrix'].astype(np.float32)
    mat = mat[:-1, :-1]
    m, s = float(mat.mean()), float(mat.std()) + 1e-8
    return (mat - m) / s


def load_dist(path):
    mat = np.load(path)['matrix'].astype(np.float32)
    m, s = float(mat.mean()), float(mat.std()) + 1e-8
    return (mat - m) / s


class HierarchicalCNN(nn.Module):
    def __init__(self, num_classes, num_context_features=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.context_mlp = nn.Sequential(nn.Linear(num_context_features, 32), nn.ReLU(), nn.Linear(32, 64))
        self.classifier = nn.Sequential(nn.Linear(256 + 64, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x, context):
        img_feat = torch.flatten(self.pool(self.features(x)), 1)
        return self.classifier(torch.cat([img_feat, self.context_mlp(context)], dim=1))


writhe_ids_df = pd.read_csv('T_level_domain_ids_writhe.csv')
proteins_df = pd.read_csv('proteins_final.csv')

parsed = proteins_df['cath_code'].apply(parse_cath)
proteins_df['C'] = parsed.apply(lambda x: x[0])
proteins_df['A'] = parsed.apply(lambda x: x[1])
proteins_df['T'] = parsed.apply(lambda x: x[2])
proteins_df = proteins_df.dropna(subset=['C', 'A', 'T']).copy()
proteins_df['C'] = proteins_df['C'].astype(int)
proteins_df['A'] = proteins_df['A'].astype(int)
proteins_df['T'] = proteins_df['T'].astype(int)

test_df = writhe_ids_df.merge(proteins_df[['domain_id', 'C', 'A', 'T']], on='domain_id', how='left')
test_df = test_df.dropna(subset=['C', 'A', 'T']).copy()

w_ckpt = torch.load('cnn_T_hierarchical_best.pth', map_location=device, weights_only=False)
w_model = HierarchicalCNN(num_classes=w_ckpt['num_T_classes']).to(device)
w_model.load_state_dict(w_ckpt['model_state'])
w_model.eval()
w_t_to_idx = w_ckpt['T_to_idx']
w_idx_to_t = w_ckpt['idx_to_T']

d_ckpt = torch.load('/home/s2147128/cnn_dist_T_best.pth', map_location=device, weights_only=False)
d_model = HierarchicalCNN(num_classes=d_ckpt['num_T_classes']).to(device)
d_model.load_state_dict(d_ckpt['model_state'])
d_model.eval()
d_t_to_idx = d_ckpt['T_to_idx']
d_idx_to_t = d_ckpt['idx_to_T']

writhe_dir = 'Writhe_Matrices_256'
dist_dir = '/storage/cmstore02/groups/TAPLab/Saniya/Distance_Matrices_256'

np.random.seed(42)
n_trials = 10

results = []

with torch.no_grad():
    for _, row in test_df.iterrows():
        domain_id = row['domain_id']
        true_c = int(row['C'])
        true_a = int(row['A'])
        true_t = int(row['T'])

        wp = os.path.join(writhe_dir, f'{domain_id}.npz')
        dp = os.path.join(dist_dir, f'{domain_id}.npz')
        if not os.path.exists(wp) or not os.path.exists(dp):
            continue

        true_t_w = w_t_to_idx.get(true_t, -1)
        true_t_d = d_t_to_idx.get(true_t, -1)
        if true_t_w == -1 or true_t_d == -1:
            continue

        c_norm = (true_c - 1) / 3.0
        a_norm = true_a / 170.0
        gt_context = torch.tensor([[c_norm, a_norm]], dtype=torch.float32).to(device)

        xw = torch.from_numpy(load_writhe(wp)).unsqueeze(0).unsqueeze(0).to(device)
        w_gt_correct = int(w_model(xw, gt_context).argmax(dim=1).item() == true_t_w)

        xd = torch.from_numpy(load_dist(dp)).unsqueeze(0).unsqueeze(0).to(device)
        d_gt_correct = int(d_model(xd, gt_context).argmax(dim=1).item() == true_t_d)

        w_corrupt_sum = 0
        d_corrupt_sum = 0

        for _ in range(n_trials):
            wrong_c = np.random.choice([c for c in [1, 2, 3, 4] if c != true_c])
            wrong_a = np.random.choice([a for a in range(10, 171, 10) if a != true_a])
            corrupt_context = torch.tensor(
                [[(wrong_c - 1) / 3.0, wrong_a / 170.0]],
                dtype=torch.float32
            ).to(device)

            w_corrupt_sum += int(w_model(xw, corrupt_context).argmax(dim=1).item() == true_t_w)
            d_corrupt_sum += int(d_model(xd, corrupt_context).argmax(dim=1).item() == true_t_d)

        results.append({
            'domain_id': domain_id,
            'true_T': true_t,
            'w_gt_correct': w_gt_correct,
            'd_gt_correct': d_gt_correct,
            'w_corrupt_acc': w_corrupt_sum / n_trials,
            'd_corrupt_acc': d_corrupt_sum / n_trials,
        })

res_df = pd.DataFrame(results)
res_df.to_csv('corrupted_context_ablation_T.csv', index=False)

print(f'proteins evaluated: {len(res_df)}')
print(f'\nground truth context:')
print(f'  writhe:   {res_df["w_gt_correct"].mean():.4f}')
print(f'  distance: {res_df["d_gt_correct"].mean():.4f}')
print(f'\ncorrupted context (avg over {n_trials} trials):')
print(f'  writhe:   {res_df["w_corrupt_acc"].mean():.4f}')
print(f'  distance: {res_df["d_corrupt_acc"].mean():.4f}')
print(f'\ndegradation:')
print(f'  writhe:   {res_df["w_gt_correct"].mean() - res_df["w_corrupt_acc"].mean():.4f}')
print(f'  distance: {res_df["d_gt_correct"].mean() - res_df["d_corrupt_acc"].mean():.4f}')
print('saved: corrupted_context_ablation_T.csv')
