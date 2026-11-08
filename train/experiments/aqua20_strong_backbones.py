# Kaggle notebook: aqua20-strong, cell 3
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== CELL 3: AQUA20 strong backbones (ConvNeXt-Small, EVA-02-Base) x (original, CLAHE) x 3 seeds + TTA (~2.5 h) =====
# 1) quick checks (~3-5 min): both new backbones train, and the eval-only / TTA path works
run_parallel(["--model convnext --recipe improved --smoke", "--model convnext --recipe improved --smoke --eval_only"],
             ["--model eva02 --recipe improved --smoke",    "--model eva02 --recipe improved --smoke --eval_only"])

# 2) TTA re-evaluation of the 8 existing improved-recipe models (no training, ~2 min each)
OLD = [f"--model {m} --variant {v} --recipe improved --eval_only" for m in ["vit", "resnet50", "densenet121", "resnet18"] for v in ["orig", "clahe"]]
NEW = lambda m, v, s: f"--model {m} --variant {v} --recipe improved --seed {s}"
run_parallel(
    OLD[:4] + [NEW("eva02", "orig", 42), NEW("eva02", "clahe", 42), NEW("eva02", "orig", 1),
               NEW("convnext", "orig", 42), NEW("convnext", "clahe", 42), NEW("convnext", "orig", 1)],
    OLD[4:] + [NEW("eva02", "clahe", 1), NEW("eva02", "orig", 2), NEW("eva02", "clahe", 2),
               NEW("convnext", "clahe", 1), NEW("convnext", "orig", 2), NEW("convnext", "clahe", 2)])

res = results("*")
res.to_csv("/kaggle/working/results_aqua20_strong.csv", index=False)
print(res.to_string())
