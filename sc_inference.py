import os
import json
import time
import torch
import yaml
from tqdm import tqdm
from datasets import load_dataset
from models.smolvlm import SmolVLM
from models.smolvlm_ft import SmolVLM_ft
from models.smolvlm_sc import SmolVLM_sc
# from models.model_name import ModelName

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_classes = {
        #"SmolVLM_ft": SmolVLM_ft,
        # "SmolVLM": SmolVLM,
        "self-consistency": SmolVLM_sc
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

        # 투표 통계 수집용
        voting_summary = {
            "total_questions": 0,
            "unanimous_3_0": 0,     # 3개 동일 (3:0)
            "majority_2_1": 0,      # 2:1 투표
            "split_1_1_1": 0,       # 1:1:1 분산
            "model_choice_used": 0   # 모델 선택 사용 횟수
        }

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
            
            # Self-consistency 모델인지 확인하고 처리
            if hasattr(model, 'select_final_answer'):
                # SmolVLM_sc인 경우: voting_stats도 함께 반환받기
                response, voting_stats = model.generate_inner_with_stats(message)
                
                # 투표 통계 업데이트
                voting_summary["total_questions"] += 1
                unique_answers = voting_stats["unique_answers"]
                max_votes = voting_stats["max_votes"]
                
                if unique_answers == 1:  # 3개 동일
                    voting_summary["unanimous_3_0"] += 1
                elif unique_answers == 2 and max_votes == 2:  # 2:1 투표
                    voting_summary["majority_2_1"] += 1
                elif unique_answers == 3:  # 1:1:1 분산
                    voting_summary["split_1_1_1"] += 1
                
                if voting_stats["selection_method"] == "model_choice":
                    voting_summary["model_choice_used"] += 1
                    
            else:
                # 일반 모델인 경우
                response = model.generate_inner(message)
                voting_stats = None
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # 결과 저장
            result_item = {
                "question_id": question_id,
                "question": question,
                "answers": answers,
                "response": response,
                "inference_time": inference_time
            }
            
            # Self-consistency 모델인 경우 추가 정보 저장
            if voting_stats:
                result_item.update({
                    "voting_stats": voting_stats,
                    "unique_answers": voting_stats["unique_answers"],
                    "max_votes": voting_stats["max_votes"],
                    "selection_method": voting_stats["selection_method"]
                })
            
            results.append(result_item)

            with open(output_filename, "w") as f:
                json.dump(results, f, indent=4)
        
        # 투표 통계 출력 및 저장
        if voting_summary["total_questions"] > 0:
            print(f"\n{'='*50}")
            print(f"VOTING STATISTICS for {model_name}")
            print(f"{'='*50}")
            print(f"Total questions: {voting_summary['total_questions']}")
            print(f"Unanimous (3:0): {voting_summary['unanimous_3_0']} ({voting_summary['unanimous_3_0']/voting_summary['total_questions']*100:.1f}%)")
            print(f"Majority (2:1): {voting_summary['majority_2_1']} ({voting_summary['majority_2_1']/voting_summary['total_questions']*100:.1f}%)")
            print(f"Split (1:1:1): {voting_summary['split_1_1_1']} ({voting_summary['split_1_1_1']/voting_summary['total_questions']*100:.1f}%)")
            print(f"Model choice used: {voting_summary['model_choice_used']} ({voting_summary['model_choice_used']/voting_summary['total_questions']*100:.1f}%)")
            print(f"{'='*50}")
            
            # 투표 통계를 별도 파일로 저장
            stats_filename = f"results/{model_name}_{dataset_name}_{dataset_split}_seed{sample_seed}_voting_stats.json"
            with open(stats_filename, "w") as f:
                json.dump(voting_summary, f, indent=4)
            print(f"Voting statistics saved to {stats_filename}")
        
        print(f"Inference complete for {model_name}. Results saved to {output_filename}")

if __name__ == "__main__":
    main()