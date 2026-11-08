# Kaggle notebook: aqua20-seeds, cell 3 (E9)
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== CELL 3: E9 extra seeds for AQUA20 (seed 1 on GPU0, seed 2 on GPU1; ~2.5-3 h) =====
def jobs(seed):
    J = []
    for m in ["vit", "resnet50", "densenet121", "resnet18"]:
        for v in ["orig", "clahe"]:
            J.append(f"--model {m} --variant {v} --recipe improved --seed {seed}")
    J += [f"--model resnet18 --recipe improved --loss focal --seed {seed}",
          f"--model resnet18 --recipe improved --sampler weighted --seed {seed}"]
    return J

run_parallel(jobs(1), jobs(2))

res = results("*_s[12]")
res.to_csv("/kaggle/working/results_E9_seeds.csv", index=False)
print(res.to_string())
