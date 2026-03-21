import os
import numpy as np
from tqdm import tqdm

IN_DIR = "Writhe_Matrices_Padded"
OUT_DIR = "Writhe_Matrices_256"
TARGET = 256
DTYPE = np.float16

os.makedirs(OUT_DIR, exist_ok=True)

def downsample_block_mean(mat, out):
    H, W = mat.shape
    newH = int(np.ceil(H / out) * out)
    newW = int(np.ceil(W / out) * out)
    padH = newH - H
    padW = newW - W
    if padH or padW:
        mat = np.pad(mat, ((0, padH), (0, padW)), mode="constant", constant_values=0.0)
    return mat.reshape(out, newH // out, out, newW // out).mean(axis=(1, 3))

files = sorted([f for f in os.listdir(IN_DIR) if f.endswith(".npz")])

for fname in tqdm(files):
    out_path = os.path.join(OUT_DIR, fname)
    if os.path.exists(out_path):
        continue
    mat = np.load(os.path.join(IN_DIR, fname))["matrix"].astype(np.float32)
    mat_ds = downsample_block_mean(mat, TARGET).astype(DTYPE)
    np.savez_compressed(out_path, matrix=mat_ds)
