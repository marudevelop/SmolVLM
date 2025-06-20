import os
import json
import time
import torch
import yaml
from tqdm import tqdm
from datasets import load_dataset
from models.smolvlm import SmolVLM
from collections import Counter

def simple_self_consistency_voting(responses):
    """
    간단한 투표 방식: 가장 많이 나온 답변 선택
    동점인 경우 첫 번째 답변 선택
    """
    counter = Counter(responses)
    most_common = counter.most_common(1)[0]
    return most_common[0], dict(counter)

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    dataset_path = "HuggingFaceM4/DocumentVQA"
    dataset_name = dataset_path.split("/")[-1]
    dataset_split = "validation"
    
    sample_size = config["dataset"]["sample_size"]
    sample_seed = config["dataset"]["sample_seed"]
    device = config["device"]

    # 기본 SmolVLM 모델 사용
    model = SmolVLM(device=device)
    
    dataset = load_dataset(dataset_path, split=dataset_split)
    if sample_size > 0:
        dataset = dataset.shuffle(seed=sample_seed).select(range(sample_size))

    if not os.path.exists("results"):
        os.makedirs("results")

    output_filename = f"results/base_self_consistency_{dataset_name}_{dataset_split}_seed{sample_seed}_test.json"
    
    results = []
    processed_qids = set()

    # 기존 결과 로드
    if os.path.exists(output_filename):
        with open(output_filename, "r") as f:
            try:
                results = json.load(f)
                processed_qids = {item["question_id"] for item in results}
                print(f"Loaded {len(results)} existing results.")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {output_filename}. Starting fresh.")

    print("Running self-consistency inference (3 runs per question)")

    # 투표 통계 수집용
    voting_summary = {
        "total_questions": 0,
        "unanimous_3_0": 0,     # 3개 동일 (3:0)
        "majority_2_1": 0,      # 2:1 투표
        "split_1_1_1": 0,       # 1:1:1 분산 (모두 다름)
    }

    for item in tqdm(dataset, desc="Self-consistency inference"):
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
        
        # 동일한 프롬프트로 3번 실행
        responses = []
        for i in range(3):
            response = model.generate_inner(message)
            responses.append(response)
        
        # 투표로 최종 답변 결정
        final_answer, vote_counts = simple_self_consistency_voting(responses)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # 투표 통계 계산
        unique_answers = len(vote_counts)
        max_votes = max(vote_counts.values())
        
        voting_summary["total_questions"] += 1
        if unique_answers == 1:  # 3개 모두 동일
            voting_summary["unanimous_3_0"] += 1
        elif unique_answers == 2 and max_votes == 2:  # 2:1 투표
            voting_summary["majority_2_1"] += 1
        elif unique_answers == 3:  # 1:1:1 분산 (모두 다름)
            voting_summary["split_1_1_1"] += 1
        
        # 결과 저장
        result_item = {
            "question_id": question_id,
            "question": question,
            "answers": answers,
            "response": final_answer,  # 최종 선택된 답변
            "all_responses": responses,  # 3개 모든 답변
            "vote_counts": vote_counts,  # 투표 결과
            "unique_answers": unique_answers,
            "max_votes": max_votes,
            "inference_time": inference_time
        }
        
        results.append(result_item)

        # 중간 저장
        with open(output_filename, "w") as f:
            json.dump(results, f, indent=4)
    
    # 투표 통계 출력 및 저장
    if voting_summary["total_questions"] > 0:
        print(f"\n{'='*50}")
        print(f"VOTING STATISTICS")
        print(f"{'='*50}")
        print(f"Total questions: {voting_summary['total_questions']}")
        print(f"Unanimous (3:0): {voting_summary['unanimous_3_0']} ({voting_summary['unanimous_3_0']/voting_summary['total_questions']*100:.1f}%)")
        print(f"Majority (2:1): {voting_summary['majority_2_1']} ({voting_summary['majority_2_1']/voting_summary['total_questions']*100:.1f}%)")
        print(f"Split (1:1:1): {voting_summary['split_1_1_1']} ({voting_summary['split_1_1_1']/voting_summary['total_questions']*100:.1f}%)")
        print(f"{'='*50}")
        
        # 투표 통계를 별도 파일로 저장
        stats_filename = f"results/self_consistency_{dataset_name}_{dataset_split}_seed{sample_seed}_voting_stats.json"
        with open(stats_filename, "w") as f:
            json.dump(voting_summary, f, indent=4)
        print(f"Voting statistics saved to {stats_filename}")
    
    print(f"Self-consistency inference complete. Results saved to {output_filename}")

if __name__ == "__main__":
    main()