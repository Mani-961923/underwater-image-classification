# ===== CELL 2: RUNNER (both GPUs) + RESULTS TABLE (reads this notebook's runs AND attached earlier runs) =====
import os, re, sys, json, glob, threading, subprocess, pandas as pd
os.makedirs("/kaggle/working/logs", exist_ok=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)

def _worker(gpu, jobs):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for job in jobs:
        log = "/kaggle/working/logs/" + re.sub(r"[^A-Za-z0-9]+", "_", job).strip("_") + ".log"
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, "/kaggle/working/train.py"] + job.split(),
                               env=env, stdout=f, stderr=subprocess.STDOUT)
        lines = open(log).read().strip().splitlines()
        if r.returncode == 0:
            print(f"[GPU{gpu}] OK     {job}\n        -> {(lines[-1] if lines else '')[:200]}", flush=True)
        else:   # show the end of the error so it can be fixed straight away
            print(f"[GPU{gpu}] FAILED {job}\n" + "\n".join("        | " + l for l in lines[-15:]), flush=True)

def run_parallel(gpu0_jobs, gpu1_jobs):
    """Each list runs one job after another on its own GPU; the two lists run at the same time."""
    ts = [threading.Thread(target=_worker, args=(0, gpu0_jobs)), threading.Thread(target=_worker, args=(1, gpu1_jobs))]
    for t in ts: t.start()
    for t in ts: t.join()
    print("All jobs finished.")

def results(pattern="*"):
    files = sorted(glob.glob(f"/kaggle/working/runs/{pattern}/metrics.json")) + \
            sorted(glob.glob(f"/kaggle/input/**/runs/{pattern}/metrics.json", recursive=True))
    rows, seen = [], set()
    for f in files:
        m = json.load(open(f))
        if m["run"] in seen:
            continue
        seen.add(m["run"])
        rows.append({"run": m["run"], "epochs": m["epochs_trained"], "sec/ep": round(m["mean_epoch_sec"]),
                     "val_macroF1": m["val"]["macro_f1"],
                     "test_acc": m["test"]["acc"], "test_macroF1": m["test"]["macro_f1"],
                     "test_weightedF1": m["test"]["weighted_f1"], "bottom5_F1": m["test"]["bottom5_f1"],
                     "clean_acc": m["test_clean"]["acc"], "clean_macroF1": m["test_clean"]["macro_f1"],
                     "tta_acc": m.get("test_tta", {}).get("acc"), "tta_macroF1": m.get("test_tta", {}).get("macro_f1")})
    df = pd.DataFrame(rows)
    return df.sort_values("run").reset_index(drop=True).round(4) if len(df) else df
