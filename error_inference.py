# 피드백 없이 틀린 정답만 사용했을 때 


import os
import json
import time
import torch
import yaml
from tqdm import tqdm
from datasets import load_dataset
#from models.smolvlm import SmolVLM
from models.model_name import SmolVLM

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_classes = {
        "SmolVLM": SmolVLM,
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
        # 결과 파일명에 "_corrected" 추가하여 구분
        output_filename = f"results/{model_name}_{dataset_name}_{dataset_split}_seed{sample_seed}_corrected.json"
        
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
        model = ModelClass(device=device) # SmolVLM 인스턴스 생성

        for item in tqdm(dataset, desc=f"Inference on {model_name}"):
            question_id = item["questionId"]
            
            if question_id in processed_qids:
                continue

            image = item["image"] 
            question = item["question"]
            answers = item["answers"] 

            # --- 첫 번째 추론 단계: 질문만 사용 ---
            # 모델 입력 메시지 구성 (텍스트만)
            message_question_only = [
                {"type": "text", "value": question}
            ]

            start_time_blind = time.time()
            blind_response = model.generate_inner(message_question_only)
            end_time_blind = time.time()
            inference_time_blind = end_time_blind - start_time_blind

            # --- 두 번째 추론 단계: 이미지, 질문, 첫 번째 답변 사용 ---
            message_with_image_and_blind_response = [
                {"type": "image", "value": image},
                {"type": "text", "value": question},
                {"type": "text_blind_answer", "value": blind_response} 
            ]

            start_time_corrected = time.time()
            corrected_response = model.generate_inner(message_with_image_and_blind_response)
            end_time_corrected = time.time()
            inference_time_corrected = end_time_corrected - start_time_corrected
            
            result_item = {
                "question_id": question_id,
                "question": question,
                "answers": answers, 
                "blind_response": blind_response, 
                "corrected_response": corrected_response, 
                "inference_time_blind": inference_time_blind,
                "inference_time_corrected": inference_time_corrected
            }
            results.append(result_item)

            with open(output_filename, "w") as f:
                json.dump(results, f, indent=4)
        
        print(f"Inference complete for {model_name}. Results saved to {output_filename}")

if __name__ == "__main__":
    main()