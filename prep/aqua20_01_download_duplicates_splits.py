# Kaggle notebook: aqua20-data-prep (3 cells)
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== CELL 1: SETUP =====
!pip install -q grad-cam ImageHash
import os, io, hashlib, pandas as pd
from huggingface_hub import snapshot_download
from datasets import load_dataset, Image as HFImage
from PIL import Image

RAW, OUT, META = "/kaggle/working/aqua20", "/kaggle/working/aqua20_img", "/kaggle/working/aqua20_meta.csv"

if not os.path.exists(RAW):                                   # step 1: download (skipped if already there)
    snapshot_download(repo_id="taufiktrf/AQUA20", repo_type="dataset", local_dir=RAW)

if not os.path.exists(META):                                  # step 2: extract to folders (skipped if done)
    ds = load_dataset(RAW)
    names = ds["train"].features["label"].names
    rows = []
    for split in ds.keys():
        dset = ds[split].cast_column("image", HFImage(decode=False))
        for i, ex in enumerate(dset):
            b = ex["image"]["bytes"]; cls = names[ex["label"]]
            img = Image.open(io.BytesIO(b))
            ext = (img.format or "PNG").lower().replace("jpeg", "jpg")
            folder = f"{OUT}/{split}/{cls}"; os.makedirs(folder, exist_ok=True)
            fp = f"{folder}/{split}_{i:05d}.{ext}"
            with open(fp, "wb") as f: f.write(b)
            rows.append({"filepath": fp, "split": split, "label": ex["label"], "class": cls,
                         "width": img.width, "height": img.height, "mode": img.mode,
                         "md5": hashlib.md5(b).hexdigest()})
    pd.DataFrame(rows).to_csv(META, index=False)

print("Ready:", len(pd.read_csv(META)), "images")


# ===== CELL 2: DUPLICATE ANALYSIS =====
import numpy as np, pandas as pd, imagehash, matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

META = "/kaggle/working/aqua20_meta.csv"
meta = pd.read_csv(META)

# ---------- 1. Exact duplicates (identical md5) ----------
dup = meta[meta.duplicated("md5", keep=False)]
g = dup.groupby("md5").agg(n=("filepath", "size"),
                           splits=("split", lambda s: "+".join(sorted(set(s)))),
                           classes=("class", lambda s: "+".join(sorted(set(s)))))
print("Exact-duplicate groups:", len(g))
print(g["splits"].value_counts().to_string())
conf = g[g["classes"].str.contains("+", regex=False)]
print("Groups with CONFLICTING labels:", len(conf))
if len(conf): print(conf.to_string())

# ---------- 2. Perceptual hashes + pairwise Hamming distances ----------
hashes = []
for fp in tqdm(meta["filepath"]):
    with Image.open(fp) as im:
        hashes.append(imagehash.phash(im.convert("RGB")))
meta["phash"] = [str(h) for h in hashes]
bits = np.array([h.hash.flatten() for h in hashes], dtype=np.float32)
ham = (64 - (bits @ bits.T + (1 - bits) @ (1 - bits).T)).round().astype(np.int16)
iu = np.triu_indices(len(meta), k=1)
dist = ham[iu]
split_arr, cls_arr = meta["split"].values, meta["class"].values

print("\nThreshold table")
for t in [0, 2, 4, 6, 8, 10]:
    m = dist <= t
    a, b = iu[0][m], iu[1][m]
    print(f"dist<={t:2d}: pairs={m.sum():6d} | train-test pairs={(split_arr[a] != split_arr[b]).sum():5d}"
          f" | different-label pairs={(cls_arr[a] != cls_arr[b]).sum():5d}")
meta.to_csv(META, index=False)

# ---------- 3. Visual check: example pairs at increasing distance ----------
rng = np.random.default_rng(0)
fig, axes = plt.subplots(5, 4, figsize=(12, 15))
for ax in axes.flat:
    ax.axis("off")
for row, t in enumerate([1, 3, 5, 7, 9]):
    idx = np.flatnonzero((dist >= t) & (dist <= t + 1))
    if len(idx) == 0:
        axes[row, 0].set_title(f"no pairs at d={t}-{t+1}", fontsize=8)
        continue
    for col, k in enumerate(rng.choice(idx, size=min(2, len(idx)), replace=False)):
        for ax, j in zip((axes[row, col * 2], axes[row, col * 2 + 1]), (iu[0][k], iu[1][k])):
            r = meta.iloc[j]
            with Image.open(r["filepath"]) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"d={dist[k]} | {r['split']} | {r['class']}", fontsize=8)
plt.tight_layout()
plt.savefig("/kaggle/working/near_dup_check.png", dpi=100)
plt.show()


# ===== CELL 3: FROZEN SPLITS + FIGURES =====
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

META   = "/kaggle/working/aqua20_meta.csv"
SPLITS = "/kaggle/working/aqua20_splits.csv"
THRESH = 6      # pHash distance <= 6 counts as "same photo"
SEED   = 42

meta = pd.read_csv(META, dtype={"phash": str})
N = len(meta)

# ---------- 1. Rebuild the 64-bit hashes and find near-duplicate pairs ----------
bits = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in meta["phash"]],
                dtype=np.float32)
ham = (64 - (bits @ bits.T + (1 - bits) @ (1 - bits).T)).round().astype(np.int16)
a, b = np.nonzero(np.triu(ham <= THRESH, k=1))

# ---------- 2. Group images that are the same photo (connected components) ----------
graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(N, N))
_, meta["group"] = connected_components(graph, directed=False)

# ---------- 3. Flag test images that also appear (near-)identically in train ----------
is_train, is_test = (meta["split"] == "train").values, (meta["split"] == "test").values
leaky = np.zeros(N, dtype=bool)
cross = is_train[a] != is_train[b]
leaky[a[cross & is_test[a]]] = True
leaky[b[cross & is_test[b]]] = True
meta["test_leak"] = leaky

# ---------- 4. Carve a validation set out of train (stratified + group-aware) ----------
tr = meta.index[is_train]
sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
_, val_pos = next(sgkf.split(tr, meta.loc[tr, "label"], groups=meta.loc[tr, "group"]))
meta["subset"] = meta["split"]
meta.loc[tr[val_pos], "subset"] = "val"

# safety checks: no photo group may be shared between train and val
g_train = set(meta.loc[meta.subset == "train", "group"])
g_val   = set(meta.loc[meta.subset == "val", "group"])
assert not (g_train & g_val), "leak between train and val!"

meta.to_csv(SPLITS, index=False)

# ---------- 5. Report ----------
print(f"Near-duplicate pairs (d<={THRESH}): {len(a)} | photo groups: {meta['group'].nunique()}")
print(f"Test images with a near-copy in train: {leaky.sum()} of {is_test.sum()} "
      f"-> clean test set = {is_test.sum() - leaky.sum()} images")
tab = meta.groupby(["class", "subset"]).size().unstack(fill_value=0)[["train", "val", "test"]]
tab["test_clean"] = meta[is_test & ~leaky].groupby("class").size().reindex(tab.index, fill_value=0)
tab["total"] = tab[["train", "val", "test"]].sum(axis=1)
print(tab.sort_values("total", ascending=False).to_string())
print("Totals:", meta["subset"].value_counts().to_dict())

# ---------- 6. Figure: class distribution (for the report) ----------
order = tab.sort_values("total").index
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(order, tab.loc[order, "total"], color="#2a78d6", height=0.7)
ax.set_xscale("log")
ax.set_xlim(10, tab["total"].max() * 3)
ax.set_xticks([10, 30, 100, 300, 1000, 3000])
ax.set_xticklabels(["10", "30", "100", "300", "1000", "3000"])
ax.minorticks_off()
for y, v in enumerate(tab.loc[order, "total"]):
    ax.text(v * 1.05, y, str(v), va="center", fontsize=8, color="#444444")
ax.set_xlabel("Number of images (log scale)")
ax.set_title("AQUA20 class distribution (train + test)", loc="left")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("/kaggle/working/fig_class_distribution.png", dpi=150); plt.show()

# ---------- 7. Figure: one sample image per class ----------
classes = sorted(meta["class"].unique())
cols = 5
rows = int(np.ceil(len(classes) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))
for ax in np.atleast_1d(axes).flat:
    ax.axis("off")
for ax, c in zip(np.atleast_1d(axes).flat, classes):
    fp = meta.loc[(meta["class"] == c) & (meta["subset"] == "train"), "filepath"].iloc[0]
    with Image.open(fp) as im:
        ax.imshow(im.convert("RGB"))
    ax.set_title(c, fontsize=9)
plt.tight_layout(); plt.savefig("/kaggle/working/fig_samples_per_class.png", dpi=150); plt.show()
