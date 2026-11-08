# Kaggle notebook: aqua20-demo, cell 1 (run interactively, GPU + Internet ON)
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== DEMO CELL 1: load the final AQUA20 model (12 networks) + sanity check (GPU, ~3-5 min) =====
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio", "grad-cam"], check=True)
import os, glob, json, cv2, numpy as np, pandas as pd, torch, timm, timm.data
from PIL import Image
from torchvision import transforms as T
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax, log_softmax
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", dev)

# ---------- 1. Find runs, data, class names ----------
RUNS = {}
for f in sorted(glob.glob("/kaggle/input/**/runs/*/metrics.json", recursive=True)):
    RUNS.setdefault(os.path.basename(os.path.dirname(f)), os.path.dirname(f))
data_csv = glob.glob("/kaggle/input/**/aqua20_data.csv", recursive=True)
assert data_csv, "aqua20_data.csv not found - attach the aqua20-clahe notebook with 'Add Input'"
DATA_BASE = os.path.dirname(data_csv[0]); D = pd.read_csv(data_csv[0])
NAMES = D.drop_duplicates("label").sort_values("label")["class"].tolist(); NUM = len(NAMES)

# The final model = pool C: ConvNeXt-S + EVA-02-B, original + CLAHE, 3 seeds, calibrated weighted voting, no TTA
POOL = [f"{m}_{v}_improved_ce_none_s{s}" for m in ("convnext", "eva02") for v in ("orig", "clahe") for s in (42, 1, 2)]
miss = [r for r in POOL if r not in RUNS or not os.path.exists(f"{RUNS[r]}/best.pt")]
assert not miss, f"Missing runs/checkpoints {miss} - attach the aqua20-strong notebook with 'Add Input'"

# ---------- 2. Re-fit temperatures + ensemble weights on VALIDATION (same code as final-aqua20) ----------
def fit_T(lg, y):
    return float(np.exp(minimize_scalar(lambda t: -log_softmax(lg / np.exp(t), 1)[np.arange(len(y)), y].mean(),
                                        bounds=(-3, 3), method="bounded").x))
def fit_weights(P, y):
    S = np.stack(P)[:, np.arange(len(y)), y]
    def f(a):
        w = softmax(a); q = w @ S + 1e-12
        gw = -(S / q).mean(1)
        return -np.log(q).mean(), w * (gw - w @ gw)
    return softmax(minimize(f, np.zeros(len(P)), jac=True, method="L-BFGS-B").x)
yv = np.load(f"{RUNS[POOL[0]]}/val_labels.npy"); yt = np.load(f"{RUNS[POOL[0]]}/test_labels.npy")
VL = [np.load(f"{RUNS[r]}/val_logits.npy") for r in POOL]; TL = [np.load(f"{RUNS[r]}/test_logits.npy") for r in POOL]
TEMPS = [fit_T(v, yv) for v in VL]
WEIGHTS = fit_weights([softmax(v / t, 1) for v, t in zip(VL, TEMPS)], yv)
p_test = np.tensordot(WEIGHTS, np.stack([softmax(t / T_, 1) for t, T_ in zip(TL, TEMPS)]), 1)
print(f"Sanity check - ensemble test accuracy from saved logits: {100 * (p_test.argmax(1) == yt).mean():.2f}% (expected 92.80%)")
for r, t, w in sorted(zip(POOL, TEMPS, WEIGHTS), key=lambda z: -z[2]):
    print(f"  {r:38s} T = {t:.3f}  weight = {w:.3f}")

# ---------- 3. Build the 12 networks and load their trained weights ----------
def build(run):
    bb = json.load(open(f"{RUNS[run]}/metrics.json"))["backbone"]
    kw = {"img_size": 224} if "eva02_base_patch14_448" in bb else {}
    m = timm.create_model(bb, pretrained=False, num_classes=NUM, **kw)
    m.load_state_dict(torch.load(f"{RUNS[run]}/best.pt", map_location="cpu", weights_only=True))
    cfg = timm.data.resolve_model_data_config(m)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    return m.to(dev).eval(), tf
MODELS = [build(r) for r in POOL]
print("Loaded", len(MODELS), "networks:", sorted({json.load(open(f'{RUNS[r]}/metrics.json'))['backbone'] for r in POOL}))

# ---------- 4. Same preprocessing as training: resize to 256 (INTER_AREA), CLAHE on LAB L channel ----------
def clahe_lab(rgb, clip=2.0, tile=8):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
def prepare(rgb):
    img = cv2.resize(np.ascontiguousarray(rgb[..., :3], dtype=np.uint8), (256, 256), interpolation=cv2.INTER_AREA)
    return img, clahe_lab(img)

@torch.no_grad()
def ensemble_probs(rgb):
    orig, enh = prepare(rgb); p = np.zeros(NUM)
    for run, (m, tf), t, w in zip(POOL, MODELS, TEMPS, WEIGHTS):
        x = tf(Image.fromarray(enh if "_clahe_" in run else orig)).unsqueeze(0).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            z = m(x).float().cpu().numpy()[0]
        p += w * softmax(z / t)
    return p, orig, enh

# ---------- 5. Check the live pipeline on 40 test images (should agree with the saved predictions) ----------
te = D[D["subset"] == "test"].reset_index(drop=True)
chk = np.random.default_rng(0).choice(len(te), 40, replace=False); agree = 0
for i in chk:
    with Image.open(f"{DATA_BASE}/{te.at[i, 'path_orig']}") as im:
        agree += int(ensemble_probs(np.array(im.convert("RGB")))[0].argmax() == p_test[i].argmax())
print(f"Live pipeline agrees with saved ensemble predictions on {agree}/40 test images (expect 38-40; tiny JPEG differences)")
