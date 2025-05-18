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
# from models.model_name import ModelName

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
device = config["device"]
dataset_name = config["dataset"]["name"]
split = config["dataset"]["split"]
sample_size = config["dataset"].get("sample_size", None)
sample_seed = config["dataset"].get("sample_seed", 42)

raw_dataset = load_dataset(dataset_name, split=split)
filtered_dataset = [item for item in raw_dataset if item["image"] is not None]
if sample_size:
    random.seed(sample_seed)
    filtered_dataset = random.sample(filtered_dataset, min(sample_size, len(filtered_dataset)))
dataset = filtered_dataset

model_classes = {
    "SmolVLM": SmolVLM,
    # "ModelName": ModelName,
}

results = []
for item in dataset:
    question = item["question"]
    choices = item["choices"]
    correct_idx = item["answer"]

    formatted_choices = []
    if choices:
        for i, choice_text in enumerate(choices):
            formatted_choices.append(f"{chr(ord('A') + i)}) {choice_text}")
        correct_answer_text = choices[correct_idx] if correct_idx < len(choices) else "N/A"
    else:
        correct_answer_text = "N/A"

    results.append({
        "question": question,
        "choices": choices,
        "formatted_choices": formatted_choices,
        "ground_truth_char": chr(ord('A') + correct_idx) if choices and correct_idx < len(choices) else "",
        "ground_truth_text": correct_answer_text,
        "answer_idx": correct_idx,
        "answers": {}
    })

os.makedirs("results_cache", exist_ok=True)
os.makedirs("results", exist_ok=True)

for model_name_key, model_class_value in model_classes.items():
    print(f"\nRunning inference with: {model_name_key}")
    sample_size_str = str(sample_size) if sample_size is not None else "all"
    result_path = f"results_cache/{model_name_key}_{dataset_name.replace('/', '_')}_{split}_{sample_seed}_{sample_size_str}.json"

    total_model_parameters = None

    if os.path.exists(result_path):
        print(f"✅ Cached result found for {model_name_key}, loading...")
        with open(result_path, "r", encoding="utf-8") as f:
            cached_results_for_model = json.load(f)
            for i, entry_data in enumerate(cached_results_for_model):
                if i < len(results):
                    if model_name_key in entry_data.get("answers", {}):
                        results[i]["answers"][model_name_key] = entry_data["answers"][model_name_key]
                        if total_model_parameters is None and "total_parameters" in entry_data["answers"][model_name_key]:
                            total_model_parameters = entry_data["answers"][model_name_key]["total_parameters"]
                    elif "pred" in entry_data and "inference_time" in entry_data:
                         results[i]["answers"][model_name_key] = {
                             "pred": entry_data["pred"],
                             "inference_time": entry_data["inference_time"]
                         }
        all_have_results = True
        if total_model_parameters is None:
            all_have_results = False
            print(f"⚠️ Model metadata (parameters) not found in cache for {model_name_key}. Re-running inference.")
        else:
            for res_entry in results:
                if model_name_key not in res_entry["answers"]:
                    all_have_results = False
                    print(f"⚠️ Cache for {model_name_key} is incomplete. Re-running inference.")
                    break
        if all_have_results:
            print(f"Model: {model_name_key}, Parameters: {total_model_parameters} (from cache)")
            continue

    model_instance = model_class_value(config['models']['smolvlm'], device)

    if total_model_parameters is None:
        total_model_parameters = sum(p.numel() for p in model_instance.model.parameters())
        print(f"Calculated - Model: {model_name_key}, Parameters: {total_model_parameters}")


    current_model_batch_results = []

    for i, item_data in enumerate(tqdm(dataset, desc=f"{model_name_key}")):
        image = item_data["image"].convert("RGB")

        current_question = item_data["question"]
        current_choices = item_data["choices"]
        current_formatted_choices_list = []
        if current_choices:
             for choice_idx, choice_text_val in enumerate(current_choices):
                current_formatted_choices_list.append(f"{chr(ord('A') + choice_idx)}) {choice_text_val}")

        current_full_question_with_choices = f"Question: {current_question}\nChoices: {', '.join(current_formatted_choices_list)}"

        if hasattr(model_instance, 'last_choices_for_parsing') and current_choices:
            model_instance.last_choices_for_parsing = current_choices

        peak_vram_MB = 0
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        start_time = time.time()
        prediction = model_instance.answer_question(image, current_full_question_with_choices)
        elapsed_time = time.time() - start_time

        if device == "cuda" and torch.cuda.is_available():
            peak_vram_MB = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

        item_model_result_data = {
            "pred": prediction,
            "inference_time": elapsed_time,
            "total_parameters": total_model_parameters,
            "peak_vram_MB": peak_vram_MB
        }

        item_for_cache = {
            "question": current_question,
            "choices": current_choices,
            "answer_idx": item_data["answer"],
            "answers": {
                model_name_key: item_model_result_data
            }
        }
        current_model_batch_results.append(item_for_cache)

        if i < len(results):
            results[i]["answers"][model_name_key] = item_model_result_data

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(current_model_batch_results, f, indent=2, ensure_ascii=False)

    del model_instance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
final_filename_path = f"results/results_{now_str}.json"
with open(final_filename_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Final results saved to: {final_filename_path}")