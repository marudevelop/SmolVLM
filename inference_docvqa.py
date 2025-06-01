import os
import json
import time
import torch
import yaml
from tqdm import tqdm
from datasets import load_dataset
from models.smolvlm import SmolVLM
# from models.model_name import ModelName

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_classes = {
        "SmolVLM": SmolVLM,
        # "ModelName": ModelName,
    }

    dataset_path = "HuggingFaceM4/DocumentVQA"
    dataset_name = dataset_path.split("/")[-1]
    dataset_split = "validation"
    
    sample_size = config["dataset"]["sample_size"]
    sample_seed = config["dataset"]["sample_seed"]

    device = config["device"]

    dataset = load_dataset(dataset_path, split=dataset_split)
    if sample_size > 0:
        dataset = dataset.shuffle(seed=sample_seed).select(range(sample_size))

    if not os.path.exists("results"):
        os.makedirs("results")

    for model_name, ModelClass in model_classes.items():
        output_filename = f"results/{model_name}_{dataset_name}_{dataset_split}_seed{sample_seed}.json"
        
        results = []
        processed_qids = set()

        if os.path.exists(output_filename):
            with open(output_filename, "r") as f:
                try:
                    results = json.load(f)
                    processed_qids = {item["question_id"] for item in results}
                    print(f"Loaded {len(results)} existing results for model {model_name}.")
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {output_filename}. Starting fresh.")

        print(f"Running inference for model: {model_name}")
        model = ModelClass(device=device)

        for item in tqdm(dataset, desc=f"Inference on {model_name}"):
            question_id = item["questionId"]
            
            if question_id in processed_qids:
                continue

            image = item["image"]
            question = item["question"]
            answers = item["answers"]

            message = [
                {"type": "image", "value": image},
                {"type": "text", "value": question}
            ]

            start_time = time.time()
            response = model.generate_inner(message)
            end_time = time.time()

            inference_time = end_time - start_time
            
            result_item = {
                "question_id": question_id,
                "question": question,
                "answers": answers,
                "response": response,
                "inference_time": inference_time
            }
            results.append(result_item)

            with open(output_filename, "w") as f:
                json.dump(results, f, indent=4)
        
        print(f"Inference complete for {model_name}. Results saved to {output_filename}")

if __name__ == "__main__":
    main()