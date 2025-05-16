import json
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np

def normalize(text):
    return text.strip().lower().rstrip("?.!,")

with open("results.json", "r") as f:
    results = json.load(f)

all_preds = defaultdict(list)
all_times = defaultdict(list)
ground_truths = []

for item in results:
    gt = normalize(item["ground_truth"])
    ground_truths.append(gt)

    for model_name, output in item["answers"].items():
        pred = normalize(output["pred"])
        all_preds[model_name].append(pred)
        all_times[model_name].append(output["inference_time"])

# 평가 결과 출력
for model_name in all_preds:
    acc = accuracy_score(ground_truths, all_preds[model_name])
    avg_time = np.mean(all_times[model_name])
    print(f"[{model_name}]")
    print(f"  - Accuracy:        {acc:.4f}")
    print(f"  - Avg Inference Time: {avg_time:.4f} sec")
    print()
