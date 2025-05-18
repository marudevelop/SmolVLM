import os
import json
from glob import glob
import numpy as np

result_files_list = sorted(glob("results/results_*.json"), reverse=True)
if not result_files_list:
    print("Error: No results_*.json file found in results/ folder.")
    exit(1)
latest_file_path = result_files_list[0]

print(f"📊 Evaluating latest file: {latest_file_path}")
with open(latest_file_path, "r", encoding="utf-8") as f:
    evaluation_results = json.load(f)

if not evaluation_results:
    print("No results to evaluate.")
    exit()

first_entry_answers = evaluation_results[0].get("answers")
if not first_entry_answers:
    print("Error: 'answers' key not found in the first result entry. Cannot determine models.")
    exit(1)
all_models = list(first_entry_answers.keys())

if not all_models:
    print("No models found in the results to evaluate.")
    exit()

accuracies_map = {name: 0 for name in all_models}
times_map = {name: [] for name in all_models}
model_total_params_map = {name: "N/A" for name in all_models}
model_peak_vram_map = {name: [] for name in all_models}

processed_entries = 0

for entry_data in evaluation_results:
    if "answer_idx" not in entry_data or not isinstance(entry_data.get("choices"), list) or not entry_data["choices"]:
        print(f"Warning: Skipping entry due to missing 'answer_idx' or 'choices'. Entry: {entry_data.get('question', 'N/A')}")
        continue

    correct_choice_idx = entry_data["answer_idx"]
    choices_list = entry_data["choices"]

    if not (0 <= correct_choice_idx < len(choices_list)):
        print(f"Warning: Invalid correct_choice_idx {correct_choice_idx} for choices length {len(choices_list)}. Entry: {entry_data.get('question', 'N/A')}")
        continue

    ground_truth_char = chr(ord('A') + correct_choice_idx)
    processed_entries += 1

    entry_answers = entry_data.get("answers", {})
    for model_name_key in all_models:
        if model_name_key in entry_answers:
            prediction_data = entry_answers[model_name_key]
            predicted_text = prediction_data.get("pred", "").strip().upper()

            predicted_char = ""
            if predicted_text:
                predicted_char = predicted_text[0]

            valid_choice_letters = [chr(ord('A') + i) for i in range(len(choices_list))]

            if predicted_char in valid_choice_letters and predicted_char == ground_truth_char:
                accuracies_map[model_name_key] += 1

            times_map[model_name_key].append(prediction_data.get("inference_time", 0.0))

            if model_total_params_map[model_name_key] == "N/A":
                model_total_params_map[model_name_key] = prediction_data.get("total_parameters", "N/A")
            
            model_peak_vram_map[model_name_key].append(prediction_data.get("peak_vram_MB", 0))
        else:
            print(f"Warning: Model '{model_name_key}' not found in answers for an entry. Question: {entry_data.get('question', 'N/A')}")


if processed_entries == 0:
    print("No valid entries were processed for evaluation.")
else:
    print(f"\nEvaluation Summary (based on {processed_entries} processed entries):")
    for model_name_val in all_models:
        accuracy = accuracies_map[model_name_val] / processed_entries if processed_entries > 0 else 0
        
        avg_time = np.mean(times_map[model_name_val]) if times_map[model_name_val] else 0.0
        
        total_params = model_total_params_map[model_name_val]
        
        avg_peak_vram = 0.0
        valid_vram_readings = [v for v in model_peak_vram_map[model_name_val] if v > 0]
        if valid_vram_readings:
            avg_peak_vram = np.mean(valid_vram_readings)
        elif model_peak_vram_map[model_name_val]:
            avg_peak_vram = 0.0
        else:
            avg_peak_vram = "N/A"

        formatted_total_params = "N/A"
        if isinstance(total_params, int):
            formatted_total_params = f"{total_params:,}"
        elif total_params != "N/A":
            try:
                formatted_total_params = f"{int(total_params):,}"
            except ValueError:
                formatted_total_params = total_params

        print(f"\n[{model_name_val}]")
        print(f"  - Parameters:           {formatted_total_params}")
        if isinstance(avg_peak_vram, float):
            print(f"  - Avg Peak VRAM:        {avg_peak_vram:.2f} MB")
        else:
            print(f"  - Avg Peak VRAM:        {avg_peak_vram}")
        print(f"  - Accuracy:             {accuracy:.4f}")
        print(f"  - Avg Inference Time:   {avg_time:.4f} sec")
    print("")