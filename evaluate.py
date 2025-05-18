import os
import json
from glob import glob

# Find latest results_*.json
result_files = sorted(glob("results/results_*.json"), reverse=True)
if not result_files:
    raise FileNotFoundError("No results_*.json file found in results/.")
latest_file = result_files[0]

print(f"📊 Evaluating latest file: {latest_file}")
with open(latest_file, "r", encoding="utf-8") as f:
    results = json.load(f)

models = list(results[0]["answers"].keys())
accuracies = {name: 0 for name in models}
times = {name: 0.0 for name in models}

for entry in results:
    ground = entry["ground_truth"].strip().lower()
    for name in models:
        pred = entry["answers"][name]["pred"].strip().lower()
        pred = pred.split("answer:")[-1] if "answer:" in pred else pred
        pred = pred.split("<")[0].strip()
        if pred == ground:
            accuracies[name] += 1
        times[name] += entry["answers"][name]["inference_time"]

n = len(results)
for name in models:
    acc = accuracies[name] / n
    avg_time = times[name] / n
    print(f"[{name}]\n  - Accuracy:        {acc:.4f}\n  - Avg Inference Time: {avg_time:.4f} sec\n")