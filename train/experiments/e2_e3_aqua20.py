# Kaggle notebook: aqua20-train-e2e3, cell 3 (E2 CLAHE + E3 imbalance)
# (Kaggle cell code exactly as committed; paths assume /kaggle/input and /kaggle/working)

# ===== CELL 3: pick the better recipe on VALIDATION, then run E2 (CLAHE) + E3 (imbalance methods) =====
e1 = results("*_orig_*_ce_none_s42")
assert len(e1) == 8, f"expected 8 E1/E1b runs, found {len(e1)} - attach the aqua20-training notebook as input"
recipe_val = e1.groupby(e1["run"].str.split("_").str[2])["val_macroF1"].mean()
BEST = recipe_val.idxmax()
print(e1.to_string())
print("\nMean VALIDATION macro-F1 by recipe:", recipe_val.round(4).to_dict(), "-> using recipe:", BEST, "\n", flush=True)

run_parallel(
    [f"--model vit --variant clahe --recipe {BEST}",
     f"--model resnet18 --variant clahe --recipe {BEST}",
     f"--model resnet18 --recipe {BEST} --loss wce"],
    [f"--model resnet50 --variant clahe --recipe {BEST}",
     f"--model densenet121 --variant clahe --recipe {BEST}",
     f"--model resnet18 --recipe {BEST} --loss focal",
     f"--model resnet18 --recipe {BEST} --sampler weighted"])

res = results("*_s42")
res.to_csv("/kaggle/working/results_E1_E2_E3.csv", index=False)
res
