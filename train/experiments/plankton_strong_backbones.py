# Kaggle notebook: plankton-strong, cell 3
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== CELL 3: PlanktonSet - TTA for existing models + CLAHE for the 4 proposal models + ConvNeXt/EVA-02 (orig + CLAHE) (~3.5 h) =====
C = "--dataset plankton --recipe improved --epochs 12 --patience 4"
EV = lambda m: f"{C} --model {m} --eval_only"
J = lambda m, v: f"{C} --model {m} --variant {v}"
run_parallel(
    [EV("vit"), EV("resnet50"), J("eva02", "orig"),  J("vit", "clahe"),         J("convnext", "clahe"), J("resnet18", "clahe")],
    [EV("densenet121"), EV("resnet18"), J("eva02", "clahe"), J("convnext", "orig"), J("densenet121", "clahe"), J("resnet50", "clahe")])
res = results("plankton_*")
res.to_csv("/kaggle/working/results_plankton_strong.csv", index=False)
print(res.to_string())
