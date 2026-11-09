# Kaggle notebook: final-figures
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== FINAL FIGURES for the report (CPU, ~1-3 min) =====
import os, glob, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score
OUT = "/kaggle/working"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e5e3"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})

def find(name):
    f = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert f, f"{name} not found - attach the notebook that produced it"
    return f[0]
RUNS = {}
for f in sorted(glob.glob("/kaggle/input/**/runs/*/metrics.json", recursive=True)):
    RUNS.setdefault(os.path.basename(os.path.dirname(f)), os.path.dirname(f))
print("Runs found:", len(RUNS))

d = pd.read_csv(find("aqua20_data.csv"))
names = d.drop_duplicates("label").sort_values("label")["class"].tolist(); NUM = len(names)
yt = d.loc[d["subset"] == "test", "label"].values
train_counts = np.bincount(d.loc[d["subset"] == "train", "label"], minlength=NUM)
def test_probs(run): return softmax(np.load(f"{RUNS[run]}/test_logits.npy"), 1)

# ---------- Figure 1: proposed vs standard baseline / published ----------
sa = pd.read_csv(find("final_aqua20_single.csv")); sig_a = pd.read_csv(find("final_aqua20_significance.csv"))
sfp = pd.read_csv(find("final_f4k_plankton_single.csv")); sig_fp = pd.read_csv(find("final_f4k_plankton_significance.csv"))
ours_single = {"AQUA20": 100 * sa[~sa["TTA"]]["acc_mean"].max(),
               "F4K": 100 * sfp[(sfp.dataset == "f4k") & (~sfp.TTA)]["acc"].max(),
               "PlanktonSet": 100 * sfp[(sfp.dataset == "plankton") & (~sfp.TTA)]["acc"].max()}
ours_final = {"AQUA20": 100 * sig_a["acc_ens"].iloc[0],
              "F4K": 100 * sig_fp[sig_fp.dataset == "f4k"]["acc_final"].iloc[0],
              "PlanktonSet": 100 * sig_fp[sig_fp.dataset == "plankton"]["acc_final"].iloc[0]}
# reference points (hard-coded): standard baseline = this work's equal soft voting / ViT with the baseline recipe;
# published = Fuad et al. (AQUA20, ConvNeXt) and Ben Tamou et al. 2022 (F4K, random image-level split)
refs = {"AQUA20":      [("Standard baseline\n(soft voting, this work)", 88.46), ("AQUA20 paper\n(ConvNeXt)", 90.69)],
        "F4K":         [("Published\n(random split)", 99.86)],
        "PlanktonSet": [("Standard baseline\n(ViT, this work)", 75.32)]}
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
for ax, k in zip(axes, ["AQUA20", "F4K", "PlanktonSet"]):
    labels = [r[0] for r in refs[k]] + ["Proposed:\nbest single model", "Proposed:\nfinal ensemble"]
    vals = [r[1] for r in refs[k]] + [ours_single[k], ours_final[k]]
    cols = (["#9b9a95", MUTED] if k == "AQUA20" else [MUTED]) + [BLUE, ORANGE]
    ax.scatter(range(len(vals)), vals, s=90, color=cols, zorder=3)          # dot plot: axis does not start at 0
    for i, v in enumerate(vals):
        ax.text(i + 0.13, v, f"{v:.2f}", ha="left", va="center", fontsize=9, color=INK)       # label to the right of the dot
    pad = 0.35 * (max(vals) - min(vals) + 1); ax.set_ylim(min(vals) - pad, min(max(vals) + pad, 100.0) if max(vals) > 99 else max(vals) + pad)
    ax.set_xlim(-0.5, len(vals) - 0.2)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels([l.replace("\n", " ") for l in labels], rotation=25, ha="right", fontsize=7.5)
    ax.set_title(k + (" (proposed: leak-free trajectory split)" if k == "F4K" else ""), fontsize=9, loc="left")
    ax.set_ylabel("Test accuracy (%)" if k == "AQUA20" else "")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_ours_vs_published.png", dpi=200); plt.show()

# ---------- Figure 2: CLAHE vs original per backbone, 3 seeds (AQUA20) ----------
B6 = ["resnet18", "resnet50", "densenet121", "vit", "convnext", "eva02"]
LAB = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "densenet121": "DenseNet-121", "vit": "ViT-B/16", "convnext": "ConvNeXt-S", "eva02": "EVA-02-B"}
fig, ax = plt.subplots(figsize=(9, 4.2))
for i, m in enumerate(B6):
    for j, (v, c) in enumerate([("orig", BLUE), ("clahe", ORANGE)]):
        accs = [100 * json.load(open(f"{RUNS[f'{m}_{v}_improved_ce_none_s{s}']}/metrics.json"))["test"]["acc"]
                for s in (42, 1, 2) if f"{m}_{v}_improved_ce_none_s{s}" in RUNS]
        xs = i + (j - 0.5) * 0.3
        ax.scatter([xs] * len(accs), accs, s=36, color=c, alpha=0.55, edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([xs - 0.11, xs + 0.11], [np.mean(accs)] * 2, color=c, linewidth=2.2, zorder=4,
                label=("Original" if v == "orig" else "CLAHE") if i == 0 else None)
ax.set_xticks(range(len(B6))); ax.set_xticklabels([LAB[m] for m in B6])
ax.set_ylabel("AQUA20 test accuracy (%)"); ax.legend(frameon=False, loc="lower right")
ax.set_title("CLAHE vs original images: each dot is one seed, bar = mean of 3 seeds", fontsize=10, loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_clahe_vs_original_seeds.png", dpi=200); plt.show()

# ---------- Figure 3: confusion matrix of the final AQUA20 model ----------
pf = np.load(find("final_aqua20_test_probs.npy")); pred_f = pf.argmax(1)
order = np.argsort(-train_counts)
cm = confusion_matrix(yt, pred_f, labels=order, normalize="true")
fig, ax = plt.subplots(figsize=(9.5, 8))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
ax.set_xticks(range(NUM)); ax.set_xticklabels([names[k] for k in order], rotation=90, fontsize=8)
ax.set_yticks(range(NUM)); ax.set_yticklabels([f"{names[k]} ({train_counts[k]})" for k in order], fontsize=8)
for r in range(NUM):
    for c in range(NUM):
        if cm[r, c] >= 0.05:
            ax.text(c, r, f"{cm[r, c]:.2f}", ha="center", va="center", fontsize=6, color="white" if cm[r, c] > 0.6 else INK)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class (training images)")
ax.set_title("Proposed final AQUA20 model: row-normalised confusion matrix (classes ordered by training size)", fontsize=10, loc="left")
plt.colorbar(im, fraction=0.04); plt.tight_layout(); plt.savefig(f"{OUT}/fig3_confusion_final_aqua20.png", dpi=200); plt.show()

# ---------- Figure 4: per-class F1, proposed final model vs ViT ----------
pv = test_probs("vit_orig_improved_ce_none_s42").argmax(1)
f_fin = f1_score(yt, pred_f, labels=order, average=None, zero_division=0)
f_vit = f1_score(yt, pv, labels=order, average=None, zero_division=0)
fig, ax = plt.subplots(figsize=(11, 4.2)); xx = np.arange(NUM)
ax.bar(xx - 0.2, f_vit, 0.4, color=MUTED, label="ViT-B/16 (improved recipe)")
ax.bar(xx + 0.2, f_fin, 0.4, color=BLUE, label="Proposed final ensemble")
ax.set_xticks(xx); ax.set_xticklabels([names[k] for k in order], rotation=60, ha="right", fontsize=8)
ax.set_ylabel("Test F1"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, loc="lower left")
ax.set_title("Per-class F1 on AQUA20 (left = most training images, right = fewest)", fontsize=10, loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_perclass_f1_aqua20.png", dpi=200); plt.show()

# ---------- Figure 5: reliability diagram (15 bins, same as the ECE in the results tables) ----------
def rel(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1); e = np.linspace(0, 1, bins + 1); xs, accs, ece = [], [], 0
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            a, c = (pred[m] == y[m]).mean(), conf[m].mean(); ece += m.mean() * abs(a - c)
            if m.sum() >= 10:                      # plot only bins with >= 10 images (tiny bins are pure noise); ECE uses all bins
                xs.append(c); accs.append(a)
    return np.array(xs), np.array(accs), ece
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
pb = test_probs("vit_orig_btech_ce_none_s42"); pi = test_probs("vit_orig_improved_ce_none_s42")
for p, c, lab in [(pb, MUTED, "ViT, baseline recipe"), (pi, YELLOW, "ViT, improved recipe (label smoothing)"),
                  (pf, BLUE, "Proposed calibrated ensemble")]:
    xs, accs, e = rel(p, yt); ax.plot(xs, accs, marker="o", markersize=5, linewidth=2, color=c, label=f"{lab} (ECE {e:.2f})")
ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Observed accuracy"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_title("Reliability diagram (AQUA20 test, 15 bins)", fontsize=10, loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_reliability_aqua20.png", dpi=200); plt.show()

# ---------- Figure 6: training curves (validation macro-F1 per epoch, seed 42, original images) ----------
fig, ax = plt.subplots(figsize=(8, 4.2)); cols = [MUTED, "#9b9a95", AQUA, YELLOW, BLUE, ORANGE]
for m, c in zip(B6, cols):
    h = pd.read_csv(f"{RUNS[f'{m}_orig_improved_ce_none_s42']}/history.csv")
    ax.plot(h["epoch"], h["val_macro_f1"], color=c, linewidth=2, label=LAB[m])
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation macro-F1"); ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower right")
ax.set_title("Training progress on AQUA20 (improved recipe, seed 42)", fontsize=10, loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_training_curves_aqua20.png", dpi=200); plt.show()
print("Saved 6 figures to /kaggle/working")

# ---------- Precision and recall (macro and weighted) for the final models and the baselines ----------
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
def prf(y, pred, n):
    out = {"acc": accuracy_score(y, pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=list(range(n)), average=avg, zero_division=0)
        out.update({f"{avg}_precision": p, f"{avg}_recall": r, f"{avg}_F1": f})
    return out
def need(run):
    if run not in RUNS:
        print(f"[missing run] {run}; similar runs:", [r for r in RUNS if run.split('_')[0] in r][:10]); return False
    return True
rows = []
# AQUA20
if all(need(f"{m}_orig_btech_ce_none_s42") for m in ["resnet18", "resnet50", "densenet121", "vit"]):
    p_base = np.mean([test_probs(f"{m}_orig_btech_ce_none_s42") for m in ["resnet18", "resnet50", "densenet121", "vit"]], 0)
    rows.append({"dataset": "AQUA20", "model": "Standard baseline (4 nets, baseline recipe, equal soft voting)", **prf(yt, p_base.argmax(1), NUM)})
rows.append({"dataset": "AQUA20", "model": "ViT-B/16 (improved recipe, seed 42)", **prf(yt, pv, NUM)})
rows.append({"dataset": "AQUA20", "model": "Proposed final model", **prf(yt, pred_f, NUM)})
# F4K and PlanktonSet
for ds, DS in [("f4k", "F4K"), ("plankton", "PlanktonSet")]:
    if not need(f"{ds}_vit_orig_improved_ce_none_s42"):
        print(f"[skip] {DS}: attach the notebook 'f4k-plankton-train' as an input"); continue
    y = np.load(f"{RUNS[f'{ds}_vit_orig_improved_ce_none_s42']}/test_labels.npy")
    p_fin = np.load(find(f"final_{ds}_test_probs.npy")); n = p_fin.shape[1]
    assert len(y) == len(p_fin), f"{DS}: label/prediction length mismatch"
    if ds == "plankton" and need("plankton_vit_orig_btech_ce_none_s42"):
        rows.append({"dataset": DS, "model": "Standard baseline (ViT-B/16, baseline recipe)", **prf(y, test_probs("plankton_vit_orig_btech_ce_none_s42").argmax(1), n)})
    rows.append({"dataset": DS, "model": "ViT-B/16 (improved recipe, seed 42)", **prf(y, test_probs(f"{ds}_vit_orig_improved_ce_none_s42").argmax(1), n)})
    rows.append({"dataset": DS, "model": "Proposed final model", **prf(y, p_fin.argmax(1), n)})
PR = pd.DataFrame(rows).round(4); PR.to_csv(f"{OUT}/precision_recall_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print("\n=== PRECISION / RECALL (test) ===\n", PR.to_string(index=False))

# per-class precision / recall of the final AQUA20 model (classes ordered by number of training images)
p_c, r_c, f_c, n_c = precision_recall_fscore_support(yt, pred_f, labels=list(order), zero_division=0)
PC = pd.DataFrame({"class": [names[k] for k in order], "train_images": train_counts[order], "test_images": n_c,
                   "precision": p_c, "recall": r_c, "F1": f_c}).round(3)
PC.to_csv(f"{OUT}/precision_recall_per_class_aqua20.csv", index=False)
print("\n=== Final AQUA20 model: per-class precision / recall ===\n", PC.to_string(index=False))
rare = list(np.argsort(train_counts)[:5])
pr5, rc5, _, _ = precision_recall_fscore_support(yt, pred_f, labels=rare, average="macro", zero_division=0)
pr5b, rc5b, _, _ = precision_recall_fscore_support(yt, p_base.argmax(1), labels=rare, average="macro", zero_division=0) if "p_base" in dir() else (np.nan, np.nan, 0, 0)
print(f"\nBottom-5 classes: precision {pr5:.3f} / recall {rc5:.3f} (final)  vs  {pr5b:.3f} / {rc5b:.3f} (standard baseline)")
