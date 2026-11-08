import argparse, glob, json, math, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

# ---------------- settings for one run ----------------
p = argparse.ArgumentParser()
p.add_argument("--dataset", default="aqua20", choices=["aqua20", "f4k", "plankton"])
p.add_argument("--split", default="subset", choices=["subset", "subset_random"])  # subset_random = leaky random split (F4K)
p.add_argument("--model", required=True, choices=["resnet18", "resnet50", "densenet121", "vit", "convnext", "eva02"])
p.add_argument("--variant", default="orig", choices=["orig", "clahe"])        # which images
p.add_argument("--recipe", default="btech", choices=["btech", "improved"])    # how to train
p.add_argument("--loss", default="ce", choices=["ce", "wce", "focal"])
p.add_argument("--sampler", default="none", choices=["none", "weighted"])
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--patience", type=int, default=6)       # early stopping on val macro-F1
p.add_argument("--bs", type=int, default=32)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--smoke", action="store_true")          # tiny quick run to test the code
p.add_argument("--out", default="/kaggle/working/runs")
p.add_argument("--eval_only", action="store_true")      # no training: load an existing run's best.pt, save TTA logits
args = p.parse_args()

prefix = "" if args.dataset == "aqua20" else f"{args.dataset}_"
suffix = "_rsplit" if args.split == "subset_random" else ""
name = prefix + f"{args.model}_{args.variant}_{args.recipe}_{args.loss}_{args.sampler}_s{args.seed}" + suffix + ("_smoke" if args.smoke else "")
base_name = name
if args.eval_only:
    name = base_name + "_ttaeval"
out = os.path.join(args.out, name)
os.makedirs(out, exist_ok=True)
if os.path.exists(f"{out}/metrics.json") and not args.smoke:
    print("Already finished, skipping:", name); raise SystemExit(0)

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = dev.type == "cuda"

# ---------------- data ----------------
found = glob.glob(f"/kaggle/input/**/{args.dataset}_data.csv", recursive=True)
assert found, f"{args.dataset}_data.csv not found - attach that dataset's prep notebook with 'Add Input'"
base = os.path.dirname(found[0])
keep = {"label", "class", "test_leak", "path_orig", "path_clahe", args.split}
df = pd.read_csv(found[0], usecols=lambda c: c in keep)
df["subset"] = df[args.split]
df["path"] = base + "/" + df[f"path_{args.variant}"]
NUM = int(df["label"].max()) + 1
class_names = df.drop_duplicates("label").sort_values("label")["class"].tolist()
tr, va, te = [df[df["subset"] == s].reset_index(drop=True) for s in ("train", "val", "test")]
if args.smoke:
    tr, va, te = tr.sample(256, random_state=0), va.sample(96, random_state=0), te.sample(96, random_state=0)
    args.epochs = min(args.epochs, 2)
counts = np.maximum(np.bincount(df.loc[df["subset"] == "train", "label"], minlength=NUM), 1)

# ---------------- model ----------------
IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
BACKBONE = ""
def build_model():
    global BACKBONE
    if args.model in ("vit", "convnext", "eva02") and not (args.model == "vit" and args.recipe == "btech"):
        import timm, timm.data
        if args.model == "vit":
            BACKBONE = "vit_base_patch16_224.augreg2_in21k_ft_in1k"
            m = timm.create_model(BACKBONE, pretrained=not args.eval_only, num_classes=NUM)
        elif args.model == "convnext":
            BACKBONE = "convnext_small.fb_in22k_ft_in1k"
            m = timm.create_model(BACKBONE, pretrained=not args.eval_only, num_classes=NUM, drop_path_rate=0.1)
        else:
            try:     # EVA-02 fine-tuned on IN-22k -> IN-1k at 448 px, position embeddings resized to 224 px
                BACKBONE = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
                m = timm.create_model(BACKBONE, pretrained=not args.eval_only, num_classes=NUM, img_size=224, drop_path_rate=0.1)
            except Exception as e:
                print("EVA-02 448 fine-tuned weights could not be loaded at 224 px:", repr(e)[:300])
                BACKBONE = "eva02_base_patch14_224.mim_in22k"
                m = timm.create_model(BACKBONE, pretrained=not args.eval_only, num_classes=NUM, drop_path_rate=0.1)
        print("Backbone:", BACKBONE, flush=True)
        cfg = timm.data.resolve_model_data_config(m)
        return m, (tuple(cfg["mean"]), tuple(cfg["std"]))
    if args.model == "vit":                                        # B.Tech recipe: torchvision ViT-B/16 (ImageNet-1k)
        BACKBONE = "torchvision vit_b_16 IMAGENET1K_V1"
        m = models.vit_b_16(weights=None if args.eval_only else "IMAGENET1K_V1")
        m.heads.head = nn.Linear(m.heads.head.in_features, NUM)
        return m, IMNET
    w = "IMAGENET1K_V1" if args.recipe == "btech" else "DEFAULT"
    BACKBONE = f"torchvision {args.model} {w}"
    m = getattr(models, args.model)(weights=None if args.eval_only else w)
    if args.model.startswith("resnet"):
        m.fc = nn.Linear(m.fc.in_features, NUM)
    else:
        m.classifier = nn.Linear(m.classifier.in_features, NUM)
    return m, IMNET

model, (MEAN, STD) = build_model()
model = model.to(dev)

# ---------------- transforms ----------------
if args.recipe == "btech":      # as described in the B.Tech report
    train_tf = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomRotation(15),
                          T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
else:                           # improved recipe: stronger augmentation
    train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.RandomRotation(15),
                          T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), T.Normalize(MEAN, STD),
                          T.RandomErasing(p=0.25)])
eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])

class ImageDS(Dataset):
    def __init__(self, d, tf):
        self.paths, self.labels, self.tf = d["path"].tolist(), d["label"].tolist(), tf
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, self.labels[i]

g = torch.Generator(); g.manual_seed(args.seed)
sampler = None
if args.sampler == "weighted":   # rare classes get drawn more often
    w = 1.0 / counts[tr["label"].values]
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(tr), replacement=True, generator=g)
lkw = dict(num_workers=args.workers, pin_memory=use_amp, persistent_workers=args.workers > 0)
train_dl = DataLoader(ImageDS(tr, train_tf), batch_size=args.bs, shuffle=sampler is None, sampler=sampler,
                      drop_last=True, generator=g, **lkw)
val_dl = DataLoader(ImageDS(va, eval_tf), batch_size=64, shuffle=False, **lkw)
test_dl = DataLoader(ImageDS(te, eval_tf), batch_size=64, shuffle=False, **lkw)

# ---------------- loss ----------------
smooth = 0.1 if args.recipe == "improved" else 0.0
class_w = torch.tensor(counts.sum() / (NUM * counts), dtype=torch.float32, device=dev)
def loss_fn(logits, y):
    logits = logits.float()
    if args.loss == "ce":
        return F.cross_entropy(logits, y, label_smoothing=smooth)
    if args.loss == "wce":
        return F.cross_entropy(logits, y, weight=class_w, label_smoothing=smooth)
    ce = F.cross_entropy(logits, y, reduction="none")        # focal loss, gamma = 2
    pt = torch.exp(-ce)
    return ((1 - pt) ** 2 * ce).mean()

# ---------------- optimiser + LR schedule ----------------
is_vit = args.model in ("vit", "eva02")
steps = args.epochs * len(train_dl)
if args.recipe == "btech":
    opt = torch.optim.Adam(model.parameters(), lr=3e-5 if is_vit else 1e-4)
    sched = None
else:
    lr = 5e-5 if is_vit else (1e-4 if args.model == "convnext" else 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    warm = min(2 * len(train_dl), steps // 2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# ---------------- helpers ----------------
@torch.no_grad()
def predict(dl, tta=False):
    model.eval(); outs = []
    for x, _ in dl:
        x = x.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            z = model(x).float()
            if tta:                                    # test-time augmentation: average with the mirror image
                z = (z + model(torch.flip(x, dims=[3])).float()) / 2
        outs.append(z.cpu())
    return torch.cat(outs).numpy()

def scores(y, logits):
    pred = logits.argmax(1)
    P, R, F1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    per = f1_score(y, pred, labels=list(range(NUM)), average=None, zero_division=0)
    order = np.argsort(counts)                       # rarest classes first (by train count)
    return {"acc": float(accuracy_score(y, pred)), "macro_p": float(P), "macro_r": float(R), "macro_f1": float(F1),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "bottom5_f1": float(per[order[:5]].mean()), "top5_f1": float(per[order[-5:]].mean()),
            "per_class_f1": {class_names[k]: float(per[k]) for k in range(NUM)}}

# ---------------- training loop (with resume + early stopping) ----------------
start_ep, best_f1, bad, hist = 0, -1.0, 0, []
if os.path.exists(f"{out}/last.pt"):
    ck = torch.load(f"{out}/last.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    if sched: sched.load_state_dict(ck["sched"])
    start_ep, best_f1, bad, hist = ck["epoch"] + 1, ck["best_f1"], ck["bad"], ck["hist"]
    print(f"Resumed from epoch {start_ep}")

if args.eval_only:
    src = sorted(glob.glob(f"/kaggle/working/runs/{base_name}/best.pt") +
                 glob.glob(f"/kaggle/input/**/runs/{base_name}/best.pt", recursive=True))
    assert src, f"best.pt for {base_name} not found (attach the notebook that trained it)"
    model.load_state_dict(torch.load(src[0], map_location=dev, weights_only=True))
    torch.save(model.state_dict(), f"{out}/best.pt")
    args.epochs = 0                                    # skip the training loop
    print("Eval-only from", src[0], flush=True)

print(f"RUN {name} | train {len(tr)} val {len(va)} test {len(te)} | device {dev}", flush=True)
for ep in range(start_ep, args.epochs):
    model.train(); t0 = time.time(); tot, n = 0.0, 0
    for x, y in train_dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if sched: sched.step()
        tot += loss.item() * len(y); n += len(y)
    ep_time = time.time() - t0
    v = scores(va["label"].values, predict(val_dl))
    hist.append({"epoch": ep + 1, "train_loss": tot / max(n, 1), "val_acc": v["acc"], "val_macro_f1": v["macro_f1"], "epoch_sec": ep_time})
    improved = v["macro_f1"] > best_f1
    if improved:
        best_f1, bad = v["macro_f1"], 0
        torch.save(model.state_dict(), f"{out}/best.pt")
    else:
        bad += 1
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "sched": sched.state_dict() if sched else None, "epoch": ep, "best_f1": best_f1, "bad": bad, "hist": hist},
               f"{out}/last.pt")
    print(f"ep {ep+1:2d} | loss {tot/max(n,1):.4f} | val acc {v['acc']:.4f} | val macroF1 {v['macro_f1']:.4f}"
          f"{' *best*' if improved else ''} | {ep_time:.0f}s", flush=True)
    if bad >= args.patience:
        print("Early stopping."); break

# ---------------- final evaluation with the best checkpoint ----------------
model.load_state_dict(torch.load(f"{out}/best.pt", map_location=dev, weights_only=True))
val_logits = predict(val_dl)
t0 = time.time(); test_logits = predict(test_dl); infer_ms = 1000 * (time.time() - t0) / len(te)
val_tta, test_tta = predict(val_dl, tta=True), predict(test_dl, tta=True)
np.save(f"{out}/val_logits_tta.npy", val_tta); np.save(f"{out}/test_logits_tta.npy", test_tta)
yt, clean = te["label"].values, ~te["test_leak"].astype(bool).values
np.save(f"{out}/val_logits.npy", val_logits); np.save(f"{out}/test_logits.npy", test_logits)
np.save(f"{out}/val_labels.npy", va["label"].values); np.save(f"{out}/test_labels.npy", yt)
np.save(f"{out}/test_clean_mask.npy", clean)
pd.DataFrame(hist).to_csv(f"{out}/history.csv", index=False)
res = {"run": name, **vars(args), "backbone": BACKBONE, "epochs_trained": len(hist), "best_val_macro_f1": best_f1,
       "mean_epoch_sec": float(np.mean([h["epoch_sec"] for h in hist])) if hist else 0.0, "infer_ms_per_img": infer_ms,
       "val": scores(va["label"].values, val_logits), "test": scores(yt, test_logits),
       "test_clean": scores(yt[clean], test_logits[clean]),
       "val_tta": scores(va["label"].values, val_tta), "test_tta": scores(yt, test_tta),
       "test_clean_tta": scores(yt[clean], test_tta[clean])}
json.dump(res, open(f"{out}/metrics.json", "w"), indent=2)
if os.path.exists(f"{out}/last.pt"):
    os.remove(f"{out}/last.pt")      # free disk; best.pt is kept
if args.eval_only:
    os.remove(f"{out}/best.pt")      # the original run already stores it
t = res["test"]
print(f"DONE {name} | test acc {t['acc']:.4f} | macroF1 {t['macro_f1']:.4f} | bottom5 F1 {t['bottom5_f1']:.4f} "
      f"| clean-test acc {res['test_clean']['acc']:.4f} | TTA acc {res['test_tta']['acc']:.4f} macroF1 {res['test_tta']['macro_f1']:.4f}", flush=True)
