# Kaggle notebook: aqua20-training, cell 3 (E1 + E1b)
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== CELL 3: E1 (B.Tech recipe) + E1b (improved recipe) on original images =====
run_parallel(
    ["--model vit --recipe btech",
     "--model resnet18 --recipe btech",
     "--model resnet18 --recipe improved",
     "--model densenet121 --recipe improved"],
    ["--model vit --recipe improved",
     "--model resnet50 --recipe btech",
     "--model densenet121 --recipe btech",
     "--model resnet50 --recipe improved"])

res = results("*_s42")                      # real runs only (smoke runs end in _smoke)
res.to_csv("/kaggle/working/results_E1.csv", index=False)
res
