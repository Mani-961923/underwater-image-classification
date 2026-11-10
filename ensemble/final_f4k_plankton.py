# Kaggle notebook: final-f4k-plankton
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== FINAL F4K + PlanktonSet ENSEMBLES: 6 backbones x (original, CLAHE), seed 42, with/without TTA (CPU, ~15-25 min) =====
import os, glob, json, itertools, numpy as np, pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax, log_softmax
from scipy.stats import binomtest
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)

RUNS = {}
for f in sorted(glob.glob("/kaggle/input/**/runs/*/metrics.json", recursive=True)):
    RUNS.setdefault(os.path.basename(os.path.dirname(f)), os.path.dirname(f))
print("Runs found:", len(RUNS))

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


B6, P4, NEW2 = ["resnet18", "resnet50", "densenet121", "vit", "convnext", "eva02"], ["resnet18", "resnet50", "densenet121", "vit"], ["convnext", "eva02"]
all_single, all_ens, all_sig = [], [], []
for DS in ["f4k", "plankton"]:
    data_csv = glob.glob(f"/kaggle/input/**/{DS}_data.csv", recursive=True)
    if not data_csv: print(f"[skip] {DS}: data csv not attached"); continue
    d0 = pd.read_csv(data_csv[0], usecols=["label", "subset"])
    NUM = int(d0["label"].max()) + 1
    counts = np.bincount(d0.loc[d0["subset"] == "train", "label"], minlength=NUM)
    RARE = np.argsort(counts)[:5]
    rn = lambda m, v: f"{DS}_{m}_{v}_improved_ce_none_s42"
    print(f"\n################ {DS.upper()} ({NUM} classes) ################")

    # ---- single models ----
    sing = []
    for m, v, tta in itertools.product(B6, ["orig", "clahe"], (False, True)):
        r = rn(m, v)
        if r not in RUNS: continue
        x = load(r, tta)
        if tta and not x["tta_used"]: continue
        s = score(softmax(x["test"], 1), x["test_labels"], x["test_clean_mask"].astype(bool))
        sing.append({"dataset": DS, "model": m, "images": v, "TTA": tta, "val_macroF1": score(softmax(x["val"], 1), x["val_labels"])["macroF1"],
                     **s, "_probs": softmax(x["test"], 1)})
    S = pd.DataFrame(sing)
    print(S.drop(columns="_probs").round(4).to_string(index=False))

    # ---- ensembles ----
    POOLS = {"A: 4 proposal models, original":          [rn(m, "orig") for m in P4],
             "B: 4 proposal models, orig+CLAHE (8)":    [rn(m, v) for m in P4 for v in ("orig", "clahe")],
             "C: ConvNeXt+EVA-02, orig+CLAHE (4)":      [rn(m, v) for m in NEW2 for v in ("orig", "clahe")],
             "D: all 6 backbones, original (6)":        [rn(m, "orig") for m in B6],
             "E: all 6 backbones, orig+CLAHE (12)":     [rn(m, v) for m in B6 for v in ("orig", "clahe")]}
    rows, store = [], {}
    for (pool, runs), tta in itertools.product(POOLS.items(), (False, True)):
        miss = [r for r in runs if r not in RUNS]
        if miss: print(f"[skip] {pool}: missing {miss[:3]}"); continue
        L = [load(r, tta) for r in runs]; yv, yt = L[0]["val_labels"], L[0]["test_labels"]; clean = L[0]["test_clean_mask"].astype(bool)
        assert all((x["val_labels"] == yv).all() and (x["test_labels"] == yt).all() for x in L)
        Ts = [fit_T(x["val"], yv) for x in L]
        Pv = [softmax(x["val"] / T, 1) for x, T in zip(L, Ts)]; Pt = [softmax(x["test"] / T, 1) for x, T in zip(L, Ts)]
        cand = {"E5 equal soft voting (raw probs)": (np.mean([softmax(x["val"], 1) for x in L], 0), np.mean([softmax(x["test"], 1) for x in L], 0)),
                "E6a soft voting + temperature scaling": (np.mean(Pv, 0), np.mean(Pt, 0))}
        wv, wt, w = weighted_voting(Pv, yv, Pt); cand["E6b weighted voting (val-learned)"] = (wv, wt)
        cand["E7b class-shared stacking"] = shared_stacking(Pv, yv, Pt)
        for meth, (pv, pt) in cand.items():
            key = f"{pool} | {'TTA' if tta else 'no TTA'} | {meth}"; store[key] = pt
            rows.append({"dataset": DS, "pool": pool, "TTA": tta, "method": meth, "val_macroF1": score(pv, yv)["macroF1"], **score(pt, yt, clean)})
        if pool.startswith("E") and tta:
            print("Weights in pool E (TTA):", sorted(zip(np.round(w, 3), [r.replace(DS + "_", "").replace("_improved_ce_none_s42", "") for r in runs]), reverse=True))
    E = pd.DataFrame(rows)
    print(E.round(4).to_string(index=False))

    # ---- final selection on validation, single comparison on test ----
    bs = S.sort_values("val_macroF1").iloc[-1]; be = E.sort_values("val_macroF1").iloc[-1]
    key = f"{be['pool']} | {'TTA' if be['TTA'] else 'no TTA'} | {be['method']}"
    pe, ps = store[key].argmax(1), bs["_probs"].argmax(1)
    vit = S[(S.model == "vit") & (S.images == "orig") & (~S.TTA)].iloc[0]["_probs"].argmax(1)
    for name, pb in [(f"best single by val ({bs['model']} {bs['images']}{' +TTA' if bs['TTA'] else ''})", ps), ("ViT original (strongest standard-baseline backbone)", vit)]:
        n01, n10, p = mcnemar(pe, pb, yt); lo, hi = boot(pe, pb, yt)
        all_sig.append({"dataset": DS, "final model": key, "vs": name, "acc_final": accuracy_score(yt, pe), "acc_other": accuracy_score(yt, pb),
                        "mF1_final": f1_score(yt, pe, average="macro", zero_division=0), "mF1_other": f1_score(yt, pb, average="macro", zero_division=0),
                        "only_final_right": n01, "only_other_right": n10, "McNemar_p": p, "mF1_diff_95CI": f"[{lo:+.3f}, {hi:+.3f}]"})
    print(f"\n=== {DS.upper()} FINAL MODEL (chosen on validation): {key} ===")
    print(f"test: acc {accuracy_score(yt, pe):.4f} | macro-F1 {f1_score(yt, pe, average='macro', zero_division=0):.4f}")
    np.save(f"/kaggle/working/final_{DS}_test_probs.npy", store[key])
    all_single.append(S.drop(columns="_probs")); all_ens.append(E)

pd.concat(all_single).round(4).to_csv("/kaggle/working/final_f4k_plankton_single.csv", index=False)
pd.concat(all_ens).round(4).to_csv("/kaggle/working/final_f4k_plankton_ensembles.csv", index=False)
sig = pd.DataFrame(all_sig).round(4); sig.to_csv("/kaggle/working/final_f4k_plankton_significance.csv", index=False)
print("\n=== FINAL vs best single / vs ViT (test) ===\n", sig.to_string(index=False))
