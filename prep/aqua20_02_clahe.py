# Kaggle notebook: aqua20-clahe
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== TASK 3: CLAHE PREPROCESSING (CPU is enough) =====
import os, glob, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

INPUT_ROOT = "/kaggle/input"
WORK       = "/kaggle/working"
SIZE       = 256          # stored size; models will see 224x224 crops/resizes of this
CLIP, TILE = 2.0, 8       # CLAHE clip limit and 8x8 tile grid (standard settings)

# ---------- 1. Find the saved data from the data-prep notebook ----------
found = glob.glob(f"{INPUT_ROOT}/**/aqua20_splits.csv", recursive=True)
assert found, "aqua20_splits.csv not found - use 'Add Input' to attach your data-prep notebook"
base = os.path.dirname(found[0])
df = pd.read_csv(found[0], dtype={"phash": str})
df["src"] = [base + p.replace("/kaggle/working", "", 1) for p in df["filepath"]]
missing = int((~df["src"].map(os.path.exists)).sum())
assert missing == 0, f"{missing} image files not found under {base}"
print("Data found at:", base, "|", len(df), "images")

# ---------- 2. Helper functions ----------
def load_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))            # also fixes the 2 RGBA images

def clahe_lab(rgb, clip=CLIP, tile=TILE):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)   # enhance lightness only
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def clahe_rgb_naive(rgb, clip=CLIP, tile=TILE):    # for comparison only: CLAHE on each colour channel
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return np.dstack([c.apply(rgb[..., k]) for k in range(3)])

def contrast_stats(rgb):
    l = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0]
    p = np.bincount(l.ravel(), minlength=256) / l.size
    p = p[p > 0]
    return float(l.std()), float(-(p * np.log2(p)).sum())   # RMS contrast, entropy (bits)

def save_jpg(rgb, path):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])

# ---------- 3. Resize once, apply CLAHE, save both versions ----------
for v in ["orig", "clahe"]:
    os.makedirs(f"{WORK}/aqua20_256/{v}", exist_ok=True)
rec = []
for i, r in enumerate(tqdm(df.itertuples(), total=len(df))):
    img = cv2.resize(load_rgb(r.src), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    enh = clahe_lab(img)
    rel_o, rel_c = f"aqua20_256/orig/{i:05d}.jpg", f"aqua20_256/clahe/{i:05d}.jpg"
    save_jpg(img, f"{WORK}/{rel_o}"); save_jpg(enh, f"{WORK}/{rel_c}")
    c0, e0 = contrast_stats(img); c1, e1 = contrast_stats(enh)
    rec.append((rel_o, rel_c, c0, c1, e0, e1))
cols = ["path_orig", "path_clahe", "contrast_orig", "contrast_clahe", "entropy_orig", "entropy_clahe"]
df = pd.concat([df, pd.DataFrame(rec, columns=cols, index=df.index)], axis=1)
df.drop(columns=["src"]).to_csv(f"{WORK}/aqua20_data.csv", index=False)

# ---------- 4. Did CLAHE change contrast? ----------
print(df[["contrast_orig", "contrast_clahe", "entropy_orig", "entropy_clahe"]].describe().round(2).to_string())
print("Images whose contrast went UP:", round(100 * (df.contrast_clahe > df.contrast_orig).mean(), 1), "%")

# ---------- 5. Figure: original vs CLAHE (LAB) vs naive per-channel CLAHE ----------
rng = np.random.default_rng(1)
all_cls = sorted(df["class"].unique())
show = rng.choice(all_cls, min(6, len(all_cls)), replace=False)
picks = [rng.choice(df.index[df["class"] == c]) for c in show]
fig, axes = plt.subplots(3, len(picks), figsize=(16, 8.4), squeeze=False)
for col, i in enumerate(picks):
    img = cv2.resize(load_rgb(df.at[i, "src"]), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    for row, (title, im) in enumerate([("Original", img), ("CLAHE on L (LAB)", clahe_lab(img)),
                                       ("CLAHE per RGB channel", clahe_rgb_naive(img))]):
        ax = axes[row, col]; ax.imshow(im); ax.axis("off")
        ax.set_title(f"{title}\n{df.at[i, 'class']}" if row == 0 else title, fontsize=9)
plt.tight_layout(); plt.savefig(f"{WORK}/fig_clahe_comparison.png", dpi=130); plt.show()
