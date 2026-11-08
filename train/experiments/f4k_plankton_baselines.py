# Kaggle notebook: f4k-plankton-train, cell 3
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== CELL 3: F4K + PlanktonSet baselines (both GPUs; ~4 h) =====
C = "--recipe improved --epochs 12 --patience 4"          # bigger datasets: fewer epochs needed
run_parallel(
    [f"--dataset f4k --model vit {C}",
     f"--dataset f4k --model resnet50 {C}",
     f"--dataset plankton --model vit {C}",
     f"--dataset plankton --model resnet50 {C}",
     "--dataset f4k --model vit --recipe btech --split subset_random --epochs 12 --patience 4"],
    [f"--dataset f4k --model densenet121 {C}",
     f"--dataset f4k --model resnet18 {C}",
     f"--dataset plankton --model densenet121 {C}",
     f"--dataset plankton --model resnet18 {C}",
     "--dataset plankton --model vit --recipe btech --epochs 12 --patience 4",
     f"--dataset f4k --model vit {C} --split subset_random"])

res = results("*")
res.to_csv("/kaggle/working/results_f4k_plankton.csv", index=False)
print(res.to_string())
