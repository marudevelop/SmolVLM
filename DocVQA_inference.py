import yaml
from datasets import load_dataset
from tqdm import tqdm
import time
import json
import random
import os
import gc
import torch
from models.smolvlm import SmolVLM
# from models.model_name import ModelName

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
device = config["device"]
dataset_name = "lmms-lab/DocVQA"  # DocVQA 데이터셋 고정
split = "validation"  # validation 셋 사용
sample_size = config["dataset"].get("sample_size", None)
sample_seed = config["dataset"].get("sample_seed", 42)

# DocVQA 데이터셋 로드
print("loading dataset...")
raw_dataset = load_dataset(dataset_name,'DocVQA', split=split)
if sample_size:
    random.seed(sample_seed)
    indices = random.sample(range(len(raw_dataset)), min(sample_size, len(raw_dataset)))
    dataset = raw_dataset.select(indices)
else:
    dataset = raw_dataset

model_classes = {
    "SmolVLM": SmolVLM,
}

def create_docvqa_cot_prompt(question):

    system_prompt = "You are a visual agent and should provide concise answers."

    media_intro_token = "Here is an image."

    media_outro_token = "Given this image,"

    instruction_for_answer = "Answer the question using a single word or phrase."

    prompt = f"{system_prompt}\n\n{media_intro_token}\n<image>\n{media_outro_token}\nQuestion: {question}\n{instruction_for_answer}\n"

    return prompt


# 기존 결과 파일 로드
result_file = "results/docvqa_results_exact2prompt.json"
existing_results = {}

if os.path.exists(result_file):
    print("Loading existing results...")
    with open(result_file, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
        # question_id를 키로 하는 딕셔너리로 변환
        for item in existing_data:
            existing_results[item["question_id"]] = item

os.makedirs("results", exist_ok=True)

for model_name_key, model_class_value in model_classes.items():
    print(f"\nRunning inference with: {model_name_key}")

    # 모델 인스턴스 생성
    model_instance = model_class_value(config['models']['smolvlm'], device)
    total_model_parameters = sum(p.numel() for p in model_instance.model.parameters())
    print(f"Model: {model_name_key}, Parameters: {total_model_parameters}")

    processed_count = 0
    skipped_count = 0

    for i, item_data in enumerate(tqdm(dataset, desc=f"{model_name_key}")):
        question_id = item_data["questionId"]
        
        # 이미 해당 question_id에 대한 결과가 있는지 확인
        if question_id in existing_results and model_name_key in existing_results[question_id].get("model_response", {}):
            skipped_count += 1
            continue

        image = item_data["image"].convert("RGB")
        question = item_data["question"]
        answers = item_data["answers"]

        # DocVQA용 프롬프트 (선택지 없이 자유 답변)
        prompt = create_docvqa_cot_prompt(question)

        # VRAM 추적 시작
        peak_vram_MB = 0
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # 추론 실행
        start_time = time.time()
        prediction = model_instance.answer_question_docvqa(image, prompt)
        elapsed_time = time.time() - start_time

        # VRAM 사용량 측정
        if device == "cuda" and torch.cuda.is_available():
            peak_vram_MB = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)

        # 결과 데이터 구성
        item_model_result_data = {
            "response": prediction,
            "inference_time": elapsed_time,
            "total_parameters": total_model_parameters,
            "peak_vram_MB": peak_vram_MB
        }

        # 기존 결과에 추가 또는 새로 생성
        if question_id in existing_results:
            existing_results[question_id]["model_response"][model_name_key] = item_model_result_data
        else:
            existing_results[question_id] = {
                "question_id": question_id,
                "question": question,
                "answers": answers,
                "model_response": {
                    model_name_key: item_model_result_data
                }
            }

        processed_count += 1

        # 주기적으로 결과 저장 (10개마다)
        if processed_count % 10 == 0:
            results_list = list(existing_results.values())
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(results_list, f, indent=2, ensure_ascii=False)

    print(f"Processed: {processed_count}, Skipped: {skipped_count}")

    # 메모리 정리
    del model_instance
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 최종 결과 저장
results_list = list(existing_results.values())
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(results_list, f, indent=2, ensure_ascii=False)
print(f"\nFinal DocVQA results saved to: {result_file}")

# 간단한 통계 출력
print(f"\nEvaluation Summary:")
print(f"- Dataset: DocVQA validation set")
print(f"- Total questions: {len(results_list)}")
print(f"- Models evaluated: {list(model_classes.keys())}")
if results_list and "model_response" in results_list[0]:
    for model_name in results_list[0]["model_response"]:
        model_results = [r["model_response"][model_name] for r in results_list if model_name in r["model_response"]]
        if model_results:
            avg_time = sum(r["inference_time"] for r in model_results) / len(model_results)
            avg_vram = sum(r["peak_vram_MB"] for r in model_results) / len(model_results)
            print(f"- {model_name}: Avg inference time: {avg_time:.3f}s, Avg VRAM: {avg_vram:.1f}MB")