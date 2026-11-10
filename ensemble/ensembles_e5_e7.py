# Kaggle notebook: ensembles
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== ENSEMBLES E5-E7 + calibration + significance (CPU, ~5-15 min) =====
import os, glob, json, numpy as np, pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax, log_softmax
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)

RUNS = {}
for f in sorted(glob.glob("/kaggle/input/**/runs/*/metrics.json", recursive=True)):
    d = os.path.dirname(f)
    RUNS.setdefault(os.path.basename(d), d)
print("Runs found:", len(RUNS))

M4 = ["resnet18", "resnet50", "densenet121", "vit"]
POOLS = {
    "AQUA20 | baseline recipe, 4 models (E1)":        ("aqua20", [f"{m}_orig_btech_ce_none_s42" for m in M4]),
    "AQUA20 | improved, 4 models (E1b)":            ("aqua20", [f"{m}_orig_improved_ce_none_s42" for m in M4]),
    "AQUA20 | improved, original + CLAHE (8)":      ("aqua20", [f"{m}_{v}_improved_ce_none_s42" for m in M4 for v in ("orig", "clahe")]),
    "F4K | improved, 4 models":                     ("f4k", [f"f4k_{m}_orig_improved_ce_none_s42" for m in M4]),
    "PlanktonSet | improved, 4 models":             ("plankton", [f"plankton_{m}_orig_improved_ce_none_s42" for m in M4]),
}

def load(run):
    d = RUNS[run]
    return {k: np.load(f"{d}/{k}.npy") for k in ["val_logits", "test_logits", "val_labels", "test_labels", "test_clean_mask"]}

def train_counts(ds, n, fallback):
    f = glob.glob(f"/kaggle/input/**/{ds}_data.csv", recursive=True)
    if f:
        d = pd.read_csv(f[0], usecols=["label", "subset"])
        return np.bincount(d.loc[d["subset"] == "train", "label"], minlength=n)
    return np.bincount(fallback, minlength=n)            # val counts as a proxy if the data csv is not attached

def ece(p, y, bins=15):
    conf, pred, e = p.max(1), p.argmax(1), 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e

def score(p, y, clean, rare):
    pred = p.argmax(1)
    per = f1_score(y, pred, labels=list(range(p.shape[1])), average=None, zero_division=0)
    return {"acc": accuracy_score(y, pred), "macroF1": f1_score(y, pred, average="macro", zero_division=0),
            "weightedF1": f1_score(y, pred, average="weighted", zero_division=0), "bottom5F1": per[rare].mean(),
            "ECE": ece(p, y), "clean_acc": accuracy_score(y[clean], pred[clean])}

def fit_temperature(logits, y):                         # temperature scaling (Guo et al., 2017), fitted on VAL only
    nll = lambda t: -log_softmax(logits / np.exp(t), 1)[np.arange(len(y)), y].mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))

def fit_weights(P, y):                                  # weights >= 0, sum to 1, minimise VAL log-loss
    S = np.stack(P)
    def nll(a):
        q = np.tensordot(softmax(a), S, 1)
        return -np.log(q[np.arange(len(y)), y] + 1e-12).mean()
    return softmax(minimize(nll, np.zeros(len(P)), method="Nelder-Mead", options={"maxiter": 2000}).x)

def weighted_voting(Pv, yv, Pt):                        # val score is OUT-OF-FOLD (weights never see the scored fold)
    Sv, St = np.stack(Pv), np.stack(Pt)
    oof = np.zeros(Sv.shape[1:])
    for tr, te in KFold(5, shuffle=True, random_state=0).split(yv):
        w = fit_weights([p[tr] for p in Pv], yv[tr])
        oof[te] = np.tensordot(w, Sv[:, te], 1)
    w = fit_weights(Pv, yv)
    return oof, np.tensordot(w, St, 1), w

def full_proba(clf, X, n):                              # map predict_proba columns back to all n classes
    out = np.full((len(X), n), 1e-6); out[:, clf.classes_] = clf.predict_proba(X)
    return out / out.sum(1, keepdims=True)

def fit_stacking(Pv, yv, Pt, n):                        # logistic-regression meta-learner trained on VAL probabilities
    Xv, Xt = np.log(np.hstack(Pv) + 1e-7), np.log(np.hstack(Pt) + 1e-7)
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6
    Xv, Xt = (Xv - mu) / sd, (Xt - mu) / sd
    best = (np.inf, None, None)
    for C in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        oof = np.zeros((len(yv), n))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(Xv):
            clf = LogisticRegression(C=C, max_iter=3000).fit(Xv[tr], yv[tr])
            oof[te] = full_proba(clf, Xv[te], n)
        loss = -np.log(oof[np.arange(len(yv)), yv]).mean()
        if loss < best[0]:
            best = (loss, C, oof)
    clf = LogisticRegression(C=best[1], max_iter=3000).fit(Xv, yv)
    return best[2], full_proba(clf, Xt, n), best[1]     # val score is OUT-OF-FOLD

def fit_shared(Xs, y, n, lam=1e-3):                   # z_c = sum_m a_m * log p_m,c + b_c   (M + n parameters)
    M = len(Xs); X = np.stack(Xs); Y = np.eye(n)[y]
    def f(theta):
        a, b = theta[:M], theta[M:]
        z = np.tensordot(a, X, 1) + b
        p = softmax(z, 1)
        loss = -(Y * log_softmax(z, 1)).sum(1).mean() + lam * (b @ b)
        g = p - Y
        ga = np.array([(g * X[m]).sum(1).mean() for m in range(M)])
        gb = g.mean(0) + 2 * lam * b
        return loss, np.r_[ga, gb]
    theta = minimize(f, np.r_[np.ones(M) / M, np.zeros(n)], jac=True, method="L-BFGS-B").x
    return theta[:M], theta[M:]

def shared_stacking(Pv, yv, Pt, n):                      # class-shared logistic stacking; val score OUT-OF-FOLD
    Xv = [np.log(p + 1e-7) for p in Pv]; Xt = [np.log(p + 1e-7) for p in Pt]
    oof = np.zeros((len(yv), n))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(yv):
        a, b = fit_shared([x[tr] for x in Xv], yv[tr], n)
        oof[te] = softmax(np.tensordot(a, np.stack([x[te] for x in Xv]), 1) + b, 1)
    a, b = fit_shared(Xv, yv, n)
    return oof, softmax(np.tensordot(a, np.stack(Xt), 1) + b, 1), a

def mcnemar(pred_a, pred_b, y):
    a, b = pred_a == y, pred_b == y
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return n01, n10, (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)

def bootstrap_diff(pred_a, pred_b, y, B=1000):
    rng, d = np.random.default_rng(0), []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], pred_a[i], average="macro", zero_division=0) -
                 f1_score(y[i], pred_b[i], average="macro", zero_division=0))
    return np.percentile(d, [2.5, 97.5])

all_rows, sig_rows = [], []
for pool, (ds, runs) in POOLS.items():
    missing = [r for r in runs if r not in RUNS]
    if missing:
        print(f"\n[skip] {pool}: missing {missing}"); continue
    L = {r: load(r) for r in runs}
    yv, yt, clean = L[runs[0]]["val_labels"], L[runs[0]]["test_labels"], L[runs[0]]["test_clean_mask"].astype(bool)
    assert all((L[r]["val_labels"] == yv).all() and (L[r]["test_labels"] == yt).all() for r in runs), "label order mismatch"
    n = L[runs[0]]["val_logits"].shape[1]
    rare = np.argsort(train_counts(ds, n, yv))[:5]
    rows = []

    Pv_raw = [softmax(L[r]["val_logits"], 1) for r in runs]; Pt_raw = [softmax(L[r]["test_logits"], 1) for r in runs]
    Ts = [fit_temperature(L[r]["val_logits"], yv) for r in runs]
    Pv_cal = [softmax(L[r]["val_logits"] / T, 1) for r, T in zip(runs, Ts)]
    Pt_cal = [softmax(L[r]["test_logits"] / T, 1) for r, T in zip(runs, Ts)]

    for r, T, pv, pt, pvc, ptc in zip(runs, Ts, Pv_raw, Pt_raw, Pv_cal, Pt_cal):
        s = score(pt, yt, clean, rare); s["ECE_after_T"] = ece(ptc, yt)
        rows.append({"method": f"single: {r}", "kind": "single", "T": round(T, 2), "val_macroF1": score(pv, yv, np.ones_like(yv, bool), rare)["macroF1"], **s})

    ens = {
        "E5 soft voting, equal weights (standard baseline)": (np.mean(Pv_raw, 0), np.mean(Pt_raw, 0), ""),
        "E6a soft voting + temperature scaling":         (np.mean(Pv_cal, 0), np.mean(Pt_cal, 0), ""),
    }
    wv, wt, w = weighted_voting(Pv_cal, yv, Pt_cal)
    ens["E6b weighted soft voting (val-optimised)"] = (wv, wt,
                                                      " ".join(f"{r.replace(ds + '_', '').split('_improved')[0].split('_btech')[0]}={x:.2f}" for r, x in zip(runs, w)))
    sv, st, C = fit_stacking(Pv_cal, yv, Pt_cal, n)
    ens["E7a stacking: full logistic regression"] = (sv, st, f"C={C}")
    hv, ht, a = shared_stacking(Pv_cal, yv, Pt_cal, n)
    ens["E7b stacking: class-shared logistic regression"] = (hv, ht, " ".join(f"{x:.2f}" for x in a))
    for name, (pv, pt, note) in ens.items():
        rows.append({"method": name, "kind": "ensemble", "T": note, "val_macroF1": score(pv, yv, np.ones_like(yv, bool), rare)["macroF1"],
                     **score(pt, yt, clean, rare)})

    tab = pd.DataFrame(rows); tab.insert(0, "pool", pool)
    all_rows.append(tab)
    print(f"\n=== {pool} | test n={len(yt)} | rare classes: {rare.tolist()} ===")
    print(tab.drop(columns=["pool"]).round(4).to_string(index=False))

    # selection on VALIDATION only: best single model vs best ensemble, compared once on TEST
    bs = tab[tab.kind == "single"].sort_values("val_macroF1").iloc[-1]
    be = tab[tab.kind == "ensemble"].sort_values("val_macroF1").iloc[-1]
    pred_s = Pt_raw[runs.index(bs["method"].replace("single: ", ""))].argmax(1)
    pred_e = ens[be["method"]][1].argmax(1)
    n01, n10, pval = mcnemar(pred_e, pred_s, yt)
    lo, hi = bootstrap_diff(pred_e, pred_s, yt)
    sig_rows.append({"pool": pool, "best_single (by val)": bs["method"].replace("single: ", ""), "best_ensemble (by val)": be["method"],
                     "acc_single": bs["acc"], "acc_ens": be["acc"], "macroF1_single": bs["macroF1"], "macroF1_ens": be["macroF1"],
                     "only_ens_right": n01, "only_single_right": n10, "McNemar_p": pval, "macroF1_diff_95CI": f"[{lo:+.3f}, {hi:+.3f}]"})

pd.concat(all_rows).round(4).to_csv("/kaggle/working/ensemble_results.csv", index=False)
sig = pd.DataFrame(sig_rows).round(4)
sig.to_csv("/kaggle/working/ensemble_significance.csv", index=False)
print("\n=== Best ensemble vs best single model (both chosen on validation; compared on test) ===")
print(sig.to_string(index=False))
