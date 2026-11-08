# Kaggle notebook: gradcam
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== E8: Grad-CAM explainability (CPU is enough, ~10-15 min) =====
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "grad-cam"], check=True)
import os, glob, numpy as np, pandas as pd, torch, torch.nn as nn, matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms as T
from scipy.special import softmax
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
torch.set_num_threads(4)
OUT = "/kaggle/working"

RUNS = {}
for f in sorted(glob.glob("/kaggle/input/**/runs/*/best.pt", recursive=True)):
    RUNS.setdefault(os.path.basename(os.path.dirname(f)), os.path.dirname(f))
print("Checkpoints found:", len(RUNS))

def data(ds):
    f = glob.glob(f"/kaggle/input/**/{ds}_data.csv", recursive=True)
    assert f, f"{ds}_data.csv not found - attach the {ds} data notebook"
    d = pd.read_csv(f[0]); base = os.path.dirname(f[0])
    te = d[d["subset"] == "test"].reset_index(drop=True)          # same order as the saved test logits
    names = d.drop_duplicates("label").sort_values("label")["class"].tolist()
    return te, base, names

def build(run, n):
    if run.split("_")[0] in ("f4k", "plankton"):
        arch = run.split("_")[1]
    else:
        arch = run.split("_")[0]
    if arch == "vit":
        import timm, timm.data
        m = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=False, num_classes=n)
        cfg = timm.data.resolve_model_data_config(m); mean, std = cfg["mean"], cfg["std"]
        layer = m.blocks[-1].norm1
        reshape = lambda t: t[:, 1:, :].reshape(t.size(0), 14, 14, t.size(2)).permute(0, 3, 1, 2)
    else:
        m = getattr(models, arch)(weights=None)
        if arch.startswith("resnet"):
            m.fc = nn.Linear(m.fc.in_features, n); layer = m.layer4[-1]
        else:
            m.classifier = nn.Linear(m.classifier.in_features, n); layer = m.features[-1]
        mean, std, reshape = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), None
    m.load_state_dict(torch.load(f"{RUNS[run]}/best.pt", map_location="cpu", weights_only=True)); m.eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean, std)])
    return m, GradCAM(model=m, target_layers=[layer], reshape_transform=reshape), tf

def cam_for(run_obj, path, cls):
    m, cam, tf = run_obj
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((224, 224))
    x = tf(rgb).unsqueeze(0)
    g = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(cls))])[0]
    return np.asarray(rgb, dtype=np.float32) / 255.0, g

def overlay(ax, rgb, g, title):
    ax.imshow(show_cam_on_image(rgb, g, use_rgb=True)); ax.set_title(title, fontsize=8); ax.axis("off")

def probs(run):
    return softmax(np.load(f"{RUNS[run]}/test_logits.npy"), 1)

# ---------------- Figure 1: AQUA20 correct vs wrong (ViT and ResNet-50, original images) ----------------
te, base, names = data("aqua20")
R_VIT, R_R50, R_VITC = "vit_orig_improved_ce_none_s42", "resnet50_orig_improved_ce_none_s42", "vit_clahe_improved_ce_none_s42"
pv = probs(R_VIT); pred = pv.argmax(1); y = te["label"].values
rng = np.random.default_rng(0)
right = rng.choice(np.flatnonzero(pred == y), 4, replace=False)
wrong = rng.choice(np.flatnonzero(pred != y), 4, replace=False)
vit, r50 = build(R_VIT, len(names)), build(R_R50, len(names))
fig, axes = plt.subplots(3, 8, figsize=(20, 8))
for c, i in enumerate(np.r_[right, wrong]):
    p = f"{base}/{te.at[i, 'path_orig']}"
    rgb, g1 = cam_for(vit, p, pred[i]); _, g2 = cam_for(r50, p, pred[i])
    axes[0, c].imshow(rgb); axes[0, c].axis("off")
    axes[0, c].set_title(f"{'CORRECT' if pred[i] == y[i] else 'WRONG'}\ntrue: {names[y[i]]}\npred: {names[pred[i]]}", fontsize=8,
                         color="green" if pred[i] == y[i] else "red")
    overlay(axes[1, c], rgb, g1, "ViT Grad-CAM"); overlay(axes[2, c], rgb, g2, "ResNet-50 Grad-CAM")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_gradcam_aqua20_correct_wrong.png", dpi=130); plt.show()

# ---------------- Figure 2: AQUA20 original vs CLAHE where the two ViTs disagree ----------------
vitc = build(R_VITC, len(names)); predc = probs(R_VITC).argmax(1)
dis = np.flatnonzero(pred != predc)
print(f"ViT original vs ViT CLAHE disagree on {len(dis)} of {len(y)} test images; "
      f"original right in {(pred[dis] == y[dis]).sum()}, CLAHE right in {(predc[dis] == y[dis]).sum()}")
pick = rng.choice(dis, min(6, len(dis)), replace=False)
fig, axes = plt.subplots(2, len(pick), figsize=(3.3 * len(pick), 7), squeeze=False)
for c, i in enumerate(pick):
    rgb_o, go = cam_for(vit, f"{base}/{te.at[i, 'path_orig']}", pred[i])
    rgb_c, gc = cam_for(vitc, f"{base}/{te.at[i, 'path_clahe']}", predc[i])
    overlay(axes[0, c], rgb_o, go, f"ORIGINAL  true: {names[y[i]]}\npred: {names[pred[i]]} {'✓' if pred[i] == y[i] else '✗'}")
    overlay(axes[1, c], rgb_c, gc, f"CLAHE\npred: {names[predc[i]]} {'✓' if predc[i] == y[i] else '✗'}")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_gradcam_aqua20_orig_vs_clahe.png", dpi=130); plt.show()

# ---------------- Figure 3 + statistic: AQUA20 rare classes (ViT) ----------------
tr_counts = pd.read_csv(glob.glob("/kaggle/input/**/aqua20_data.csv", recursive=True)[0]).query("subset == 'train'")["label"].value_counts()
rare = tr_counts.sort_values().index[:5].tolist()
idx = [rng.choice(np.flatnonzero(y == k)) for k in rare]
fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
for ax, i in zip(axes, idx):
    rgb, g = cam_for(vit, f"{base}/{te.at[i, 'path_orig']}", pred[i])
    overlay(ax, rgb, g, f"RARE  true: {names[y[i]]}\npred: {names[pred[i]]} {'✓' if pred[i] == y[i] else '✗'}")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_gradcam_aqua20_rare.png", dpi=130); plt.show()

# ---------------- Figure 4 + statistic: F4K - does the model look at the burned-in timestamp? ----------------
te4, base4, names4 = data("f4k")
R4 = "f4k_resnet50_orig_improved_ce_none_s42"
if R4 in RUNS:
    r50f = build(R4, len(names4)); p4 = probs(R4).argmax(1); y4 = te4["label"].values
    sample = rng.choice(len(te4), 200, replace=False)
    top_share = []
    for i in sample:
        _, g = cam_for(r50f, f"{base4}/{te4.at[i, 'path_orig']}", p4[i])
        top_share.append(g[:27].sum() / (g.sum() + 1e-8))              # top 12% of rows (where timestamps appear)
    top_share = np.array(top_share)
    print(f"F4K ResNet-50: mean share of Grad-CAM mass in the top 12% of the image = {top_share.mean():.3f} "
          f"(0.12 = no preference); images with > 30% of mass there: {(top_share > 0.30).sum()} / {len(sample)}")
    show = sample[np.argsort(-top_share)[:4]].tolist() + sample[np.argsort(top_share)[:4]].tolist()
    fig, axes = plt.subplots(1, 8, figsize=(20, 3.6))
    for ax, i in zip(axes, show):
        rgb, g = cam_for(r50f, f"{base4}/{te4.at[i, 'path_orig']}", p4[i])
        overlay(ax, rgb, g, f"true: {names4[y4[i]][:18]}\npred: {names4[p4[i]][:18]} {'✓' if p4[i] == y4[i] else '✗'}")
    plt.suptitle("F4K ResNet-50: 4 images with MOST attention at the top (left) vs 4 with LEAST (right)", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_gradcam_f4k_timestamp.png", dpi=130); plt.show()
else:
    print("[skip] F4K checkpoint not found - attach f4k-plankton-train")
print("Done. Figures saved in /kaggle/working")
