# Kaggle notebook: f4k-data-prep (cell 1 = shared helpers, cell 2 = F4K)
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


# ===== F4K: download, trajectory-grouped splits, resize + CLAHE  (CPU notebook, Internet ON) =====
import tarfile, urllib.request
URL = "https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/GROUNDTRUTH/RECOG/Archive/fishRecognition_GT.tar"
TMP = "/tmp/f4k"                                 # raw files kept OUT of /kaggle/working (not saved)
SPECIES = ["Dascyllus reticulatus", "Plectroglyphidodon dickii", "Chromis chrysura", "Amphiprion clarkii",
           "Chaetodon lunulatus", "Chaetodon trifascialis", "Myripristis kuntee", "Acanthurus nigrofuscus",
           "Hemigymnus fasciatus", "Neoniphon sammara", "Abudefduf vaigiensis", "Canthigaster valentini",
           "Pomacentrus moluccensis", "Zebrasoma scopas", "Hemigymnus melapterus", "Lutjanus fulvus",
           "Scolopsis bilineata", "Scaridae", "Pempheris vanicolensis", "Zanclus cornutus",
           "Neoglyphidodon nigroris", "Balistapus undulatus", "Siganus fuscescens"]

if not os.path.isdir(TMP):
    os.makedirs(TMP)
    tar_path = f"{TMP}/f4k.tar"
    print("Downloading (~510 MB)..."); urllib.request.urlretrieve(URL, tar_path)
    with tarfile.open(tar_path) as t:
        t.extractall(TMP, members=[m for m in t.getmembers() if "mask" not in m.name.lower()], filter="data")
    os.remove(tar_path)

rows = []
for fp in glob.glob(f"{TMP}/**/*.*", recursive=True):
    parts = fp.replace("\\", "/").split("/")
    cls_dir = [p for p in parts[:-1] if re.fullmatch(r"fish_\d{1,2}", p)]
    if not cls_dir or "mask" in fp.lower() or not fp.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    k = int(cls_dir[-1].split("_")[1]) - 1
    nums = re.findall(r"\d+", os.path.splitext(parts[-1])[0])
    with open(fp, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    rows.append({"src": fp, "label": k, "class": SPECIES[k], "track": nums[0], "md5": md5})
df = pd.DataFrame(rows).sort_values("src").reset_index(drop=True)
print("Images:", len(df), "| classes:", df["label"].nunique(), "| trajectories:", df["track"].nunique(),
      "| exact duplicate files:", int(df.duplicated("md5").sum()))
assert len(df) > 0, "no images found - send the folder listing: !find /tmp/f4k -maxdepth 3 | head"

df["group"] = make_groups([df["track"].values, df["md5"].values])   # same trajectory OR identical file
df["subset"] = group_split(df["label"].values, df["group"].values)
df["subset_random"] = random_split(df["label"].values)
df["test_leak"] = False
tr_g = set(df.loc[df.subset_random == "train", "group"])
leak = (df.subset_random == "test") & df["group"].isin(tr_g)
print(f"RANDOM split: {leak.sum()} of {(df.subset_random == 'test').sum()} test images "
      f"({100 * leak.mean() / (df.subset_random == 'test').mean():.1f}%) share a trajectory with training images")

def load_f4k(i):
    with Image.open(df.at[i, "src"]) as im:
        return np.array(im.convert("RGB"))
df = process_all(df, "f4k", load_f4k)
df.drop(columns=["src"]).to_csv(f"{WORK}/f4k_data.csv", index=False)
report(df, "f4k")
