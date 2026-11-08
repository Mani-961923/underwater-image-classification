# Kaggle notebook: f4k-strong, cell 3
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== CELL 3: F4K - TTA for existing models + CLAHE for the 4 proposal models + ConvNeXt/EVA-02 (orig + CLAHE) (~3.5 h) =====
C = "--dataset f4k --recipe improved --epochs 12 --patience 4"
EV = lambda m: f"{C} --model {m} --eval_only"
J = lambda m, v: f"{C} --model {m} --variant {v}"
run_parallel(
    [EV("vit"), EV("resnet50"), J("eva02", "orig"),  J("vit", "clahe"),         J("convnext", "clahe"), J("resnet18", "clahe")],
    [EV("densenet121"), EV("resnet18"), J("eva02", "clahe"), J("convnext", "orig"), J("densenet121", "clahe"), J("resnet50", "clahe")])
res = results("f4k_*")
res.to_csv("/kaggle/working/results_f4k_strong.csv", index=False)
print(res.to_string())
