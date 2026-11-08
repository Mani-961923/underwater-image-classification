# Kaggle notebook: final-aqua20
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== FINAL AQUA20 ENSEMBLE: 6 backbones x (original, CLAHE) x 3 seeds, with/without TTA (CPU, ~10-20 min) =====
import os, glob, json, itertools, numpy as np, pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax, log_softmax
from scipy.stats import binomtest
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)
DS = "aqua20"

RUNS = {}
for f in sorted(glob.glob("/kaggle/input/**/runs/*/metrics.json", recursive=True)):
    RUNS.setdefault(os.path.basename(os.path.dirname(f)), os.path.dirname(f))
print("Runs found:", len(RUNS))

def run_name(m, v, s): return f"{m}_{v}_improved_ce_none_s{s}"
def load(run, tta):
    d = RUNS[run]; out = {k: np.load(f"{d}/{k}.npy") for k in ["val_labels", "test_labels", "test_clean_mask"]}
    src = None
    if tta:
        if os.path.exists(f"{d}/val_logits_tta.npy"): src = (d, "_tta")
        elif run + "_ttaeval" in RUNS:               src = (RUNS[run + "_ttaeval"], "_tta")
    if src is None: src = (d, "")
    out["val"] = np.load(f"{src[0]}/val_logits{src[1]}.npy"); out["test"] = np.load(f"{src[0]}/test_logits{src[1]}.npy")
    out["tta_used"] = src[1] == "_tta"
    return out

data_csv = glob.glob(f"/kaggle/input/**/{DS}_data.csv", recursive=True)
d0 = pd.read_csv(data_csv[0], usecols=["label", "subset", "class"])
NUM = int(d0["label"].max()) + 1
counts = np.bincount(d0.loc[d0["subset"] == "train", "label"], minlength=NUM)
RARE = np.argsort(counts)[:5]

def ece(p, y, bins=15):
    conf, pred, e = p.max(1), p.argmax(1), 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e
def score(p, y, clean=None):
    pred = p.argmax(1); per = f1_score(y, pred, labels=list(range(NUM)), average=None, zero_division=0)
    s = {"acc": accuracy_score(y, pred), "macroF1": f1_score(y, pred, average="macro", zero_division=0),
         "weightedF1": f1_score(y, pred, average="weighted", zero_division=0), "bottom5F1": per[RARE].mean(), "ECE": ece(p, y)}
    if clean is not None: s["clean_acc"] = accuracy_score(y[clean], pred[clean])
    return s
def fit_T(lg, y):
    return float(np.exp(minimize_scalar(lambda t: -log_softmax(lg / np.exp(t), 1)[np.arange(len(y)), y].mean(),
                                        bounds=(-3, 3), method="bounded").x))
def fit_weights(P, y):                       # simplex weights minimising val log-loss (analytic gradient, L-BFGS)
    S = np.stack(P)[:, np.arange(len(y)), y]                    # (M, n) prob of the true class
    def f(a):
        w = softmax(a); q = w @ S + 1e-12
        gw = -(S / q).mean(1)
        return -np.log(q).mean(), w * (gw - w @ gw)
    return softmax(minimize(f, np.zeros(len(P)), jac=True, method="L-BFGS-B").x)
def weighted_voting(Pv, yv, Pt):
    Sv = np.stack(Pv); oof = np.zeros(Sv.shape[1:])
    for tr, te in KFold(5, shuffle=True, random_state=0).split(yv):
        oof[te] = np.tensordot(fit_weights([p[tr] for p in Pv], yv[tr]), Sv[:, te], 1)
    w = fit_weights(Pv, yv); return oof, np.tensordot(w, np.stack(Pt), 1), w
def fit_shared(Xs, y, lam=1e-3):
    M = len(Xs); X = np.stack(Xs); Y = np.eye(NUM)[y]
    def f(th):
        a, b = th[:M], th[M:]; z = np.tensordot(a, X, 1) + b; p = softmax(z, 1); g = p - Y
        return (-(Y * log_softmax(z, 1)).sum(1).mean() + lam * b @ b,
                np.r_[[(g * X[m]).sum(1).mean() for m in range(M)], g.mean(0) + 2 * lam * b])
    th = minimize(f, np.r_[np.ones(M) / M, np.zeros(NUM)], jac=True, method="L-BFGS-B").x
    return th[:M], th[M:]
def shared_stacking(Pv, yv, Pt):
    Xv = [np.log(p + 1e-7) for p in Pv]; Xt = [np.log(p + 1e-7) for p in Pt]; oof = np.zeros((len(yv), NUM))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(yv):
        a, b = fit_shared([x[tr] for x in Xv], yv[tr]); oof[te] = softmax(np.tensordot(a, np.stack([x[te] for x in Xv]), 1) + b, 1)
    a, b = fit_shared(Xv, yv); return oof, softmax(np.tensordot(a, np.stack(Xt), 1) + b, 1)
def mcnemar(pa, pb, y):
    a, b = pa == y, pb == y; n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return n01, n10, (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)
def boot(pa, pb, y, B=2000):
    rng = np.random.default_rng(0); d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], pa[i], average="macro", zero_division=0) - f1_score(y[i], pb[i], average="macro", zero_division=0))
    return np.percentile(d, [2.5, 97.5])

# ---------------- 1. Single models: mean +- sd over 3 seeds ----------------
B6 = ["resnet18", "resnet50", "densenet121", "vit", "convnext", "eva02"]
rows = []
for m, v in itertools.product(B6, ["orig", "clahe"]):
    rs = [run_name(m, v, s) for s in (42, 1, 2) if run_name(m, v, s) in RUNS]
    for tta in (False, True):
        sc = []
        for r in rs:
            L = load(r, tta); sc.append(score(softmax(L["test"], 1), L["test_labels"], L["test_clean_mask"].astype(bool)))
        if sc:
            df = pd.DataFrame(sc)
            rows.append({"model": m, "images": v, "TTA": tta, "seeds": len(sc),
                         **{f"{k}_mean": df[k].mean() for k in ["acc", "macroF1", "bottom5F1", "clean_acc"]},
                         "acc_sd": df["acc"].std(ddof=1) if len(sc) > 1 else np.nan, "macroF1_sd": df["macroF1"].std(ddof=1) if len(sc) > 1 else np.nan})
single = pd.DataFrame(rows).round(4)
print("\n=== Single models (test), mean over seeds ===\n", single.to_string(index=False))

# ---------------- 2. Ensembles ----------------
P4, NEW2 = ["resnet18", "resnet50", "densenet121", "vit"], ["convnext", "eva02"]
POOLS = {
    "A: 4 proposal models, original, seed 42":            [run_name(m, "orig", 42) for m in P4],
    "B: 4 proposal models, orig+CLAHE, 3 seeds (24)":     [run_name(m, v, s) for m in P4 for v in ("orig", "clahe") for s in (42, 1, 2)],
    "C: ConvNeXt+EVA-02, orig+CLAHE, 3 seeds (12)":       [run_name(m, v, s) for m in NEW2 for v in ("orig", "clahe") for s in (42, 1, 2)],
    "D: all 6 backbones, original only, 3 seeds (18)":    [run_name(m, "orig", s) for m in B6 for s in (42, 1, 2)],
    "E: all 6 backbones, orig+CLAHE, 3 seeds (36)":       [run_name(m, v, s) for m in B6 for v in ("orig", "clahe") for s in (42, 1, 2)],
}
ens_rows, store = [], {}
for (pool, runs), tta in itertools.product(POOLS.items(), (False, True)):
    miss = [r for r in runs if r not in RUNS]
    if miss: print(f"[skip] {pool}: missing {miss[:3]}..."); continue
    L = [load(r, tta) for r in runs]; yv, yt = L[0]["val_labels"], L[0]["test_labels"]; clean = L[0]["test_clean_mask"].astype(bool)
    assert all((x["val_labels"] == yv).all() and (x["test_labels"] == yt).all() for x in L)
    Ts = [fit_T(x["val"], yv) for x in L]
    Pv = [softmax(x["val"] / T, 1) for x, T in zip(L, Ts)]; Pt = [softmax(x["test"] / T, 1) for x, T in zip(L, Ts)]
    cand = {"E5 equal soft voting (raw probs)": (np.mean([softmax(x["val"], 1) for x in L], 0), np.mean([softmax(x["test"], 1) for x in L], 0)),
            "E6a soft voting + temperature scaling": (np.mean(Pv, 0), np.mean(Pt, 0))}
    wv, wt, w = weighted_voting(Pv, yv, Pt); cand["E6b weighted voting (val-learned)"] = (wv, wt)
    cand["E7b class-shared stacking"] = shared_stacking(Pv, yv, Pt)
    for meth, (pv, pt) in cand.items():
        key = f"{pool} | {'TTA' if tta else 'no TTA'} | {meth}"
        store[key] = pt
        ens_rows.append({"pool": pool, "TTA": tta, "method": meth, "val_macroF1": score(pv, yv)["macroF1"], **score(pt, yt, clean)})
    if "E: all 6" in pool and tta:
        print("\nWeights in pool E (TTA), largest 10:", sorted(zip(np.round(w, 3), runs), reverse=True)[:10])
ens = pd.DataFrame(ens_rows).round(4)
print("\n=== Ensembles (val macro-F1 is out-of-fold; test reported once) ===\n", ens.to_string(index=False))

# ---------------- 3. Final model selection on VALIDATION, compared on TEST ----------------
Lref = load(run_name("vit", "orig", 42), False); yt = Lref["test_labels"]; yv = Lref["val_labels"]
sing = []
for r in [r for r in RUNS if r.endswith(("_s42", "_s1", "_s2")) and "_improved_ce_none_" in r and r.split("_")[0] in B6]:
    for tta in (False, True):
        x = load(r, tta)
        if tta and not x["tta_used"]: continue
        sing.append((score(softmax(x["val"], 1), yv)["macroF1"], r + (" +TTA" if tta else ""), softmax(x["test"], 1)))
best_single = max(sing, key=lambda z: z[0])
best_ens = ens.sort_values("val_macroF1").iloc[-1]
key = f"{best_ens['pool']} | {'TTA' if best_ens['TTA'] else 'no TTA'} | {best_ens['method']}"
pe, ps = store[key].argmax(1), best_single[2].argmax(1)
vit_b = softmax(load(run_name("vit", "orig", 42), False)["test"], 1).argmax(1)
summary = []
for name, pb in [(f"best single model by val ({best_single[1]})", ps), ("ViT original seed 42 (B.Tech's best architecture)", vit_b)]:
    n01, n10, p = mcnemar(pe, pb, yt); lo, hi = boot(pe, pb, yt)
    summary.append({"final ensemble vs": name, "acc_ens": accuracy_score(yt, pe), "acc_other": accuracy_score(yt, pb),
                    "mF1_ens": f1_score(yt, pe, average="macro"), "mF1_other": f1_score(yt, pb, average="macro"),
                    "only_ens_right": n01, "only_other_right": n10, "McNemar_p": p, "mF1_diff_95CI": f"[{lo:+.3f}, {hi:+.3f}]"})
summary = pd.DataFrame(summary).round(4)
print(f"\n=== FINAL MODEL (chosen on validation): {key} ===")
print(f"test: acc {accuracy_score(yt, pe):.4f} | macro-F1 {f1_score(yt, pe, average='macro'):.4f} | "
      f"clean acc {accuracy_score(yt[Lref['test_clean_mask'].astype(bool)], pe[Lref['test_clean_mask'].astype(bool)]):.4f}")
print(summary.to_string(index=False))
single.to_csv("/kaggle/working/final_aqua20_single.csv", index=False); ens.to_csv("/kaggle/working/final_aqua20_ensembles.csv", index=False)
summary.to_csv("/kaggle/working/final_aqua20_significance.csv", index=False)
np.save("/kaggle/working/final_aqua20_test_probs.npy", store[key])
