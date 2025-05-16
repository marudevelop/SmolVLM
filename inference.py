import yaml
from datasets import load_dataset
from tqdm import tqdm
import time
import json

from models.smolvlm import SmolVLM
from models.smolvlm_cot import SmolVLMCoT
from models.smolvlm_selfconsistency import SmolVLMSelfConsistency

# Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

device = config["device"]
dataset_name = config["dataset"]["name"]
split = config["dataset"]["split"]

# Load dataset
dataset = load_dataset(dataset_name)[split]
# 빠른 실험을 위해 Validation 데이터셋에서 랜덤으로 100개만 사용 (시드 고정)
dataset = dataset.shuffle(seed=42).select(range(100))

# Initialize models
models = {
    "SmolVLM": SmolVLM(config['models']['smolvlm'], device),
    # "SmolVLM_CoT": SmolVLMCoT(config['models']['smolvlm'], device),
    # "SmolVLM_SelfConsistency": SmolVLMSelfConsistency(config['models']['smolvlm'], device),
}

# Inference
results = []
for item in tqdm(dataset, desc="Running inference"):
    if item["image"] is None:
        continue
    image = item["image"].convert("RGB")
    question = item["question"]
    choices = item["choices"]
    correct_idx = item["answer"]
    correct_answer = choices[correct_idx]

    full_question = f"{question} Choices: {', '.join(choices)}"

    model_outputs = {}
    for name, model in models.items():
        start = time.time()
        pred = model.answer_question(image, full_question)
        elapsed = time.time() - start
        model_outputs[name] = {
            "pred": pred,
            "inference_time": elapsed
        }

    results.append({
        "question": question,
        "choices": choices,
        "ground_truth": correct_answer,
        "answers": model_outputs
    })

# Save
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
