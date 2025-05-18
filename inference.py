import yaml
from datasets import load_dataset
from tqdm import tqdm
import time
import json
import random
import os
import datetime
import gc
import torch
from models.smolvlm import SmolVLM
from models.model_name import ModelName

# Load config
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
device = config["device"]
dataset_name = config["dataset"]["name"]
split = config["dataset"]["split"]
sample_size = config["dataset"].get("sample_size", None)
sample_seed = config["dataset"].get("sample_seed", 42)

# Load and filter dataset
raw_dataset = load_dataset(dataset_name)[split]
filtered_dataset = [item for item in raw_dataset if item["image"] is not None]
if sample_size:
    random.seed(sample_seed)
    filtered_dataset = random.sample(filtered_dataset, min(sample_size, len(filtered_dataset)))
dataset = filtered_dataset

# Model classes
model_classes = {
    "SmolVLM": SmolVLM,
    "Model_Name": ModelName,
}

# Initialize result skeleton
results = []
for item in dataset:
    image = item["image"].convert("RGB")
    question = item["question"]
    choices = item["choices"]
    correct_idx = item["answer"]
    correct_answer = choices[correct_idx]
    full_question = f"{question} Choices: {', '.join(choices)}"
    results.append({
        "question": question,
        "choices": choices,
        "ground_truth": correct_answer,
        "answers": {}
    })

# Run inference per model
os.makedirs("results_cache", exist_ok=True)
os.makedirs("results", exist_ok=True)

for model_name, model_class in model_classes.items():
    print(f"\nRunning inference with: {model_name}")
    result_path = f"results_cache/{model_name}_{dataset_name.replace('/', '_')}_{split}_{sample_seed}_{sample_size}.json"
    if os.path.exists(result_path):
        print(f"✅ Cached result found for {model_name}, loading...")
        with open(result_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
            for i, entry in enumerate(cached):
                results[i]["answers"][model_name] = entry["answers"][model_name]
        continue

    model = model_class(config['models']['smolvlm'], device)
    model_results = []
    for i, item in enumerate(tqdm(dataset, desc=f"{model_name}")):
        image = item["image"].convert("RGB")
        question = item["question"]
        choices = item["choices"]
        full_question = f"{question} Choices: {', '.join(choices)}"
        start = time.time()
        pred = model.answer_question(image, full_question)
        elapsed = time.time() - start
        results[i]["answers"][model_name] = {
            "pred": pred,
            "inference_time": elapsed
        }
        model_results.append(results[i])

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=2, ensure_ascii=False)

    # Clear memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

# Save all results
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
final_filename = f"results/results_{now}.json"
with open(final_filename, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Final results saved to: {final_filename}")