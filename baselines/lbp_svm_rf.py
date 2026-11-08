# Kaggle notebook: lbp-baselines
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== E0: LBP + SVM / Random Forest classical baselines (CPU, ~30-60 min for all 3 datasets) =====
import os, glob, time, numpy as np, pandas as pd
from PIL import Image
from joblib import Parallel, delayed
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
pd.set_option("display.width", 250)

P, R, SIZE = 8, 1, 128                      # 8 neighbours, radius 1, images 128x128 (as in the B.Tech report)
GRID = 4                                    # 4x4 spatial grid of LBP histograms (adds "where" to "what texture")

def lbp_features(path):
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((SIZE, SIZE)), dtype=np.uint8)
    codes = local_binary_pattern(g, P, R, method="uniform")            # uniform LBP -> values 0..P+1
    nb = P + 2
    feats = [np.bincount(codes.astype(int).ravel(), minlength=nb) / codes.size]      # global histogram
    s = SIZE // GRID
    for r in range(GRID):
        for c in range(GRID):
            blk = codes[r*s:(r+1)*s, c*s:(c+1)*s].astype(int).ravel()
            feats.append(np.bincount(blk, minlength=nb) / blk.size)
    return np.concatenate(feats)                                         # (1 + 16) * 10 = 170 features

def scores(y, pred, counts, clean):
    per = f1_score(y, pred, labels=list(range(len(counts))), average=None, zero_division=0)
    rare = np.argsort(counts)[:5]
    return {"acc": accuracy_score(y, pred), "macroF1": f1_score(y, pred, average="macro", zero_division=0),
            "weightedF1": f1_score(y, pred, average="weighted", zero_division=0), "bottom5F1": per[rare].mean(),
            "clean_acc": accuracy_score(y[clean], pred[clean])}

rows = []
for ds in ["aqua20", "f4k", "plankton"]:
    f = glob.glob(f"/kaggle/input/**/{ds}_data.csv", recursive=True)
    if not f:
        print(f"[skip] {ds}: data csv not attached"); continue
    d = pd.read_csv(f[0]); base = os.path.dirname(f[0])
    d = d[d["subset"].isin(["train", "test"])].reset_index(drop=True)
    t0 = time.time()
    X = np.array(Parallel(n_jobs=-1, batch_size=256)(delayed(lbp_features)(f"{base}/{p}") for p in d["path_orig"]))
    print(f"{ds}: LBP features {X.shape} in {time.time() - t0:.0f}s", flush=True)
    tr, te = (d["subset"] == "train").values, (d["subset"] == "test").values
    y = d["label"].values; n = int(y.max()) + 1
    counts = np.bincount(y[tr], minlength=n); clean = ~d.loc[te, "test_leak"].astype(bool).values
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    for name, clf in [("LBP + SVM (RBF)", SVC(kernel="rbf", C=10, gamma="scale", cache_size=2000)),
                      ("LBP + Random Forest (200 trees)", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))]:
        t0 = time.time(); clf.fit(Xtr, y[tr]); pred = clf.predict(Xte)
        r = {"dataset": ds, "model": name, "train_n": int(tr.sum()), "test_n": int(te.sum()),
             **scores(y[te], pred, counts, clean), "fit_sec": round(time.time() - t0)}
        rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)

res = pd.DataFrame(rows).round(4)
res.to_csv("/kaggle/working/results_E0_lbp.csv", index=False)
print("\n", res.to_string(index=False))
