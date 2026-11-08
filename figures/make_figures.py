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

# ---------- Figure 1: proposed vs existing / published ----------
sa = pd.read_csv(find("final_aqua20_single.csv")); sig_a = pd.read_csv(find("final_aqua20_significance.csv"))
sfp = pd.read_csv(find("final_f4k_plankton_single.csv")); sig_fp = pd.read_csv(find("final_f4k_plankton_significance.csv"))
ours_single = {"AQUA20": 100 * sa[~sa["TTA"]]["acc_mean"].max(),
               "F4K": 100 * sfp[(sfp.dataset == "f4k") & (~sfp.TTA)]["acc"].max(),
               "PlanktonSet": 100 * sfp[(sfp.dataset == "plankton") & (~sfp.TTA)]["acc"].max()}
ours_final = {"AQUA20": 100 * sig_a["acc_ens"].iloc[0],
              "F4K": 100 * sig_fp[sig_fp.dataset == "f4k"]["acc_final"].iloc[0],
              "PlanktonSet": 100 * sig_fp[sig_fp.dataset == "plankton"]["acc_final"].iloc[0]}
existing = {"AQUA20": 90.0, "F4K": 99.0, "PlanktonSet": 75.0}      # existing model: best reported accuracy per dataset
ds = list(existing)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, k in zip(axes, ds):
    vals = [existing[k], ours_single[k], ours_final[k]]
    labels = ["Existing model (reported)", "Proposed: best single model", "Proposed: final ensemble"]
    if k == "AQUA20":
        vals.insert(1, 90.69); labels.insert(1, "AQUA20 paper (ConvNeXt)")
    cols = [MUTED, "#9b9a95", BLUE, ORANGE][-len(vals):] if k == "AQUA20" else [MUTED, BLUE, ORANGE]
    ax.scatter(range(len(vals)), vals, s=90, color=cols, zorder=3)          # dot plot: axis does not start at 0
    for i, v in enumerate(vals):
        ax.text(i, v + 0.12 * (max(vals) - min(vals) + 1), f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    pad = 0.35 * (max(vals) - min(vals) + 1); ax.set_ylim(min(vals) - pad, max(vals) + pad); ax.set_xlim(-0.6, len(vals) - 0.4)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title(k + (" (proposed model: leak-free trajectory split)" if k == "F4K" else ""), fontsize=9, loc="left")
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

# ---------- Figure 4: per-class F1, proposed final model vs existing model (ViT) ----------
pv = test_probs("vit_orig_improved_ce_none_s42").argmax(1)
f_fin = f1_score(yt, pred_f, labels=order, average=None, zero_division=0)
f_vit = f1_score(yt, pv, labels=order, average=None, zero_division=0)
fig, ax = plt.subplots(figsize=(11, 4.2)); xx = np.arange(NUM)
ax.bar(xx - 0.2, f_vit, 0.4, color=MUTED, label="Existing model (ViT-B/16)")
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
for p, c, lab in [(pb, MUTED, "Existing model (ViT, original recipe)"), (pi, YELLOW, "ViT, improved recipe (label smoothing)"),
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
