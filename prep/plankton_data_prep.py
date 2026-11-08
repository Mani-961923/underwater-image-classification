# Kaggle notebook: plankton-data-prep (cell 1 = shared helpers, cell 2 = PlanktonSet)
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ---------- shared helpers: grouping, leak-free splits, resize + CLAHE ----------
import os, re, io, glob, hashlib, cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

WORK, SIZE, SEED, CLIP, TILE = "/kaggle/working", 256, 42, 2.0, 8

def make_groups(keys_list):
    """Images that share ANY key (e.g. same trajectory or identical file) end up in the same group."""
    n = len(keys_list[0]); rows, cols, offset = [], [], n
    for keys in keys_list:
        codes, uniq = pd.factorize(pd.Series(keys))
        rows += list(range(n)); cols += list(codes + offset); offset += len(uniq)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(offset, offset))
    return connected_components(g, directed=False)[1][:n]

def group_split(label, group, seed=SEED, frac=1/7):
    """Per-class group split: ~1/7 test, ~1/7 val, rest train (by image count).
    Every class with >= 3 groups gets at least one group in EACH subset; no group is shared between subsets."""
    label, group = np.asarray(label), np.asarray(group)
    g = pd.DataFrame({"label": label, "group": group})
    g_label = g.groupby("group")["label"].agg(lambda s: s.mode().iloc[0])     # class of each group
    g_size = g.groupby("group").size()
    rng = np.random.default_rng(seed)
    assign = {}
    for k in np.unique(g_label.values):
        gs = g_label.index[g_label.values == k].to_numpy().copy()
        rng.shuffle(gs)
        target = frac * g_size[gs].sum()
        guarantee = len(gs) >= 3
        i = 0
        for sub, keep_after in [("test", 2 if guarantee else 1), ("val", 1)]:
            count = 0
            while i < len(gs) and (len(gs) - i - 1) >= keep_after and \
                    ((guarantee and count == 0) or count < target):
                assign[gs[i]] = sub; count += g_size[gs[i]]; i += 1
        for x in gs[i:]:
            assign[x] = "train"
    sub = np.array([assign[x] for x in group], dtype=object)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(group[sub == a]) & set(group[sub == b]), f"group shared between {a} and {b}"
    return sub

def random_split(label, seed=SEED):
    """Ordinary stratified random split (ignores groups) - used only to MEASURE leakage."""
    idx = np.arange(len(label)); sub = np.array(["train"] * len(label), dtype=object)
    rest, te = train_test_split(idx, test_size=1/7, stratify=label, random_state=seed)
    tr, va = train_test_split(rest, test_size=1/6, stratify=label[rest], random_state=seed)
    sub[te], sub[va] = "test", "val"
    return sub

def to_square(img, pad_value=None):
    if pad_value is None:
        return img
    h, w = img.shape[:2]; s = max(h, w)
    out = np.full((s, s) + img.shape[2:], pad_value, dtype=img.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return out

def clahe_rgb(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(TILE, TILE)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def clahe_gray(gray):
    return cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(TILE, TILE)).apply(gray)

def contrast(gray):
    p = np.bincount(gray.ravel(), minlength=256) / gray.size; p = p[p > 0]
    return float(gray.std()), float(-(p * np.log2(p)).sum())

def process_all(df, name, load_fn, gray=False, pad_value=None):
    """Resize every image to SIZE x SIZE once, save original + CLAHE versions as JPEG."""
    for v in ["orig", "clahe"]:
        os.makedirs(f"{WORK}/{name}_256/{v}", exist_ok=True)
    rec = []
    for i in tqdm(range(len(df))):
        img = to_square(load_fn(i), pad_value)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if gray:
            enh = clahe_gray(img); g0, g1 = img, enh
            img, enh = np.dstack([img] * 3), np.dstack([enh] * 3)
        else:
            enh = clahe_rgb(img)
            g0, g1 = [cv2.cvtColor(x, cv2.COLOR_RGB2LAB)[..., 0] for x in (img, enh)]
        ro, rc = f"{name}_256/orig/{i:06d}.jpg", f"{name}_256/clahe/{i:06d}.jpg"
        for arr, rel in [(img, ro), (enh, rc)]:
            cv2.imwrite(f"{WORK}/{rel}", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        (c0, e0), (c1, e1) = contrast(g0), contrast(g1)
        rec.append((ro, rc, c0, c1, e0, e1))
    cols = ["path_orig", "path_clahe", "contrast_orig", "contrast_clahe", "entropy_orig", "entropy_clahe"]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rec, columns=cols)], axis=1)

def report(df, name):
    tab = df.groupby(["class", "subset"]).size().unstack(fill_value=0).reindex(columns=["train", "val", "test"], fill_value=0)
    tab["total"] = tab.sum(axis=1)
    print(tab.sort_values("total", ascending=False).to_string())
    print("Totals:", df["subset"].value_counts().to_dict())
    empty = tab.index[(tab[["train", "val", "test"]] == 0).any(axis=1)].tolist()
    print("Classes missing from some subset:", empty if empty else "none")
    print(df[["contrast_orig", "contrast_clahe"]].describe().round(2).to_string())
    print("Contrast went UP for", round(100 * (df.contrast_clahe > df.contrast_orig).mean(), 1), "% of images")
    rng = np.random.default_rng(1)
    picks = rng.choice(len(df), min(6, len(df)), replace=False)
    fig, axes = plt.subplots(2, len(picks), figsize=(16, 6), squeeze=False)
    for c, i in enumerate(picks):
        for r, col in enumerate(["path_orig", "path_clahe"]):
            ax = axes[r, c]; ax.axis("off")
            with Image.open(f"{WORK}/{df.at[i, col]}") as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(("Original\n" + str(df.at[i, "class"])[:22]) if r == 0 else "CLAHE", fontsize=9)
    plt.tight_layout(); plt.savefig(f"{WORK}/fig_{name}_clahe.png", dpi=120); plt.show()


# ===== PlanktonSet 1.0 (NDSB 2015 labelled train set): find data, duplicate-grouped splits, pad + resize + CLAHE =====
import zipfile
blobs, rows = [], []

def from_folder(root):                     # root/<class_name>/<image>.jpg
    for c in sorted(os.listdir(root)):
        for fp in sorted(glob.glob(os.path.join(root, c, "*.jpg"))):
            with open(fp, "rb") as f:
                yield c, f.read()

def from_zip(path):                        # zip containing .../train/<class_name>/<image>.jpg
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            p = n.replace("\\", "/").split("/")
            if n.lower().endswith(".jpg") and len(p) >= 2 and "train" in p[:-2]:
                yield p[-2], z.read(n)

source = None
for d in glob.glob("/kaggle/input/**/train", recursive=True):              # 1) Kaggle competition data, unzipped
    if os.path.isdir(d) and sum(os.path.isdir(os.path.join(d, s)) for s in os.listdir(d)) >= 100:
        source = ("folder", d); break
if source is None:
    z = glob.glob("/kaggle/input/**/train.zip", recursive=True)             # 2) Kaggle competition data, zipped
    if z:
        source = ("zip", z[0])

if source is not None:
    print("Using Kaggle competition data:", source)
    pairs = from_folder(source[1]) if source[0] == "folder" else from_zip(source[1])
    for c, b in tqdm(pairs):
        blobs.append(b); rows.append({"class": c, "md5": hashlib.md5(b).hexdigest()})
    df = pd.DataFrame(rows)
    names = sorted(df["class"].unique())
    df["label"] = df["class"].map({c: k for k, c in enumerate(names)})
else:                                                                         # 3) Hugging Face (needs HF_TOKEN secret)
    from datasets import load_dataset, Image as HFImage
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
    ds = load_dataset("project-oceania/planktonset1.0", split="train", token=token)
    names = ds.features["label"].names
    ds = ds.cast_column("image", HFImage(decode=False))
    for ex in tqdm(ds, total=len(ds)):
        b = ex["image"]["bytes"]; blobs.append(b)
        rows.append({"label": ex["label"], "class": names[ex["label"]], "md5": hashlib.md5(b).hexdigest()})
    df = pd.DataFrame(rows)

assert len(df) > 0 and df["label"].nunique() >= 100, f"plankton data not found or incomplete ({len(df)} images)"
dup = df.duplicated("md5", keep=False)
conflict = df[dup].groupby("md5")["label"].nunique().gt(1).sum()
print("Images:", len(df), "| classes:", df["label"].nunique(),
      "| exact duplicate files:", int(df.duplicated("md5").sum()), "| duplicate groups with conflicting labels:", int(conflict))

df["group"] = make_groups([df["md5"].values])            # identical files always stay together
df["subset"] = group_split(df["label"].values, df["group"].values)
df["test_leak"] = False

def load_plankton(i):
    with Image.open(io.BytesIO(blobs[i])) as im:
        return np.array(im.convert("L"))
df = process_all(df, "plankton", load_plankton, gray=True, pad_value=255)   # white padding keeps shape/aspect ratio
df.to_csv(f"{WORK}/plankton_data.csv", index=False)
report(df, "plankton")
