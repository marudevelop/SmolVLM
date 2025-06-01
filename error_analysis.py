import json
import numpy as np 
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image
import os 


try:
    from evaluate_docvqa import (
        process_answer, 
        levenshtein_distance, 
        anls_compute,
    )
except ImportError:
    print("오류: 'evaluate_docvqa.py'를 찾을 수 없습니다. ANLS 계산에 필요합니다.")
    print("스크립트와 동일한 디렉토리에 있거나 PYTHONPATH에 설정되어 있는지 확인하세요.")
    def anls_compute(gt, pred):
        print("경고: 실제 anls_compute 함수가 로드되지 않았습니다. 임시 값 0.5를 반환합니다.")
        return 0.5
    
    


RESULTS_FILE_PATH = "results/SmolVLM_DocumentVQA_validation_seed42.json"
ANALYSIS_OUTPUT_DIR = "low_score_analysis_output" # 낮은 점수 샘플 분석 결과를 저장할 디렉토리
NUM_SAMPLES_TO_ANALYZE = 100 # 낮은 점수 샘플의 수
DOCVQA_DATASET_NAME = "HuggingFaceM4/DocumentVQA"
DOCVQA_SPLIT = "validation" 

def analyze_low_score_samples():
    # 출력 디렉토리 설정
    IMAGE_SAVE_DIR = os.path.join(ANALYSIS_OUTPUT_DIR, "images")
    METADATA_FILE_PATH = os.path.join(ANALYSIS_OUTPUT_DIR, "metadata.jsonl")

    # 1. JSON 파일에서 결과 로드
    try:
        with open(RESULTS_FILE_PATH, "r", encoding="utf-8") as f:
            all_results_data = json.load(f)
    except FileNotFoundError:
        print(f"오류: 결과 파일을 찾을 수 없습니다. 경로: {RESULTS_FILE_PATH}")
        return
    except json.JSONDecodeError:
        print(f"오류: {RESULTS_FILE_PATH} 파일이 올바른 JSON 형식이 아닙니다.")
        return

    # 2. 각 항목에 대해 ANLS 점수 계산
    scored_results = []
    print("ANLS 점수 계산 중...")
    for item in tqdm(all_results_data, desc="ANLS 점수 계산"):
        question_id = item.get("question_id")
        predicted_raw = item.get("response")
        ground_truths_raw = item.get("answers")

        if question_id is None or predicted_raw is None or ground_truths_raw is None:
            print(f"항목 건너뜀 (필수 필드 누락): {item.get('question_id', 'ID 없음')}")
            continue
        predicted_str = str(predicted_raw)
        if not ground_truths_raw:
            min_anls_distance = 1.0
        else:
            ground_truths_str_list = [str(gt) for gt in ground_truths_raw]
            anls_distances_for_gts = [anls_compute(gt_str, predicted_str) for gt_str in ground_truths_str_list]
            min_anls_distance = min(anls_distances_for_gts) if anls_distances_for_gts else 1.0
        item_score = 1.0 - min_anls_distance
        scored_results.append({
            "question_id": question_id,
            "score": item_score,
            "original_item": item
        })

    scored_results.sort(key=lambda x: x["score"])
    low_score_items_to_analyze = scored_results[:NUM_SAMPLES_TO_ANALYZE]
    
    if not low_score_items_to_analyze:
        print("분석할 낮은 점수의 항목이 없습니다.")
        return

    print(f"\n점수가 가장 낮은 상위 {len(low_score_items_to_analyze)}개의 항목을 분석하고 이미지 및 메타데이터를 저장합니다.")

    # 분석 결과 저장 디렉토리 생성
    try:
        os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
        os.makedirs(IMAGE_SAVE_DIR, exist_ok=True) # 이미지 저장 하위 디렉토리 생성
        print(f"분석 결과는 '{ANALYSIS_OUTPUT_DIR}' 디렉토리에 저장됩니다.")
        print(f"이미지는 '{IMAGE_SAVE_DIR}'에, 메타데이터는 '{METADATA_FILE_PATH}'에 저장됩니다.")
    except OSError as e:
        print(f"오류: 출력 디렉토리 생성 실패 ('{ANALYSIS_OUTPUT_DIR}' 또는 '{IMAGE_SAVE_DIR}'): {e}")
        return

    # 5. DocumentVQA 데이터셋 로드
    print(f"{DOCVQA_DATASET_NAME} 데이터셋 ({DOCVQA_SPLIT} 스플릿) 로드 중...")
    try:
        docvqa_dataset = load_dataset(DOCVQA_DATASET_NAME, split=DOCVQA_SPLIT, trust_remote_code=True)
    except Exception as e:
        print(f"HuggingFace 데이터셋 ({DOCVQA_DATASET_NAME}) 로드 중 오류 발생: {e}")
        print("데이터셋 없이는 이미지 및 전체 메타데이터에 접근할 수 없습니다. 스크립트를 종료합니다.")
        return
        
    print("데이터셋 인덱싱 중...")
    dataset_dict = {sample['questionId']: sample for sample in tqdm(docvqa_dataset, desc="데이터셋 인덱싱")}

    # 6 & 7. 데이터 필터링, 매칭, 정보 저장 (메타데이터 파일 및 이미지)
    print("\n--- 낮은 점수 항목 상세 정보 저장 중 ---")
    with open(METADATA_FILE_PATH, "w", encoding="utf-8") as metadata_f:
        for i, scored_item in enumerate(tqdm(low_score_items_to_analyze, desc="낮은 점수 항목 처리 및 저장")):
            q_id = scored_item["question_id"]
            model_response = scored_item["original_item"]["response"]
            ground_truths_from_results = scored_item["original_item"]["answers"]
            original_question_text_from_json = scored_item["original_item"]["question"]
            score = scored_item["score"]

            # 각 샘플에 대한 데이터 레코드 준비
            sample_data_record = {
                "question_id": q_id,
                "anls_score": score,
                "model_prediction": model_response,
                "model_input_question": original_question_text_from_json,
                "model_input_answers": ground_truths_from_results,
                "image_path": None, # 이미지가 저장되면 경로가 채워짐

                # 원본 데이터셋의 필드 (샘플 발견 시 채워짐)
                "dataset_question": None,
                "dataset_question_types": None,
                "dataset_doc_id": None,
                "dataset_ucsf_document_id": None,
                "dataset_ucsf_document_page_no": None,
                "dataset_answers": None,
            }
            
            # 터미널에 기본 정보 출력 (선택 사항)
            # print(f"\n{i+1}. Question ID: {q_id} (ANLS Score: {score:.4f})")
            # print(f"   🤖 모델 응답: {model_response}")


            dataset_sample = dataset_dict.get(q_id)
            current_image_saved_path = None

            if dataset_sample:
                sample_data_record.update({
                    "dataset_question": dataset_sample.get("question"),
                    "dataset_question_types": dataset_sample.get("question_types"),
                    "dataset_doc_id": dataset_sample.get("docId"),
                    "dataset_ucsf_document_id": dataset_sample.get("ucsf_document_id"),
                    "dataset_ucsf_document_page_no": dataset_sample.get("ucsf_document_page_no"),
                    "dataset_answers": dataset_sample.get("answers"),
                })

                original_image = dataset_sample.get("image")
                if original_image and isinstance(original_image, Image.Image):
                    safe_q_id_str = str(q_id).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                    image_filename = f"image_{safe_q_id_str}.png" 
                    image_save_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
                    
                    try:
                        original_image.save(image_save_path)
                        current_image_saved_path = image_save_path
                    except Exception as e:
                        print(f"   🖼️ 이미지 저장 중 오류 발생 (ID: {q_id}, 경로: {image_save_path}): {e}")
            else:
                print(f"   ⚠️ 경고: DocumentVQA 데이터셋에서 Question ID '{q_id}'를 찾을 수 없습니다. 해당 ID의 데이터셋 정보는 누락됩니다.")
            
            sample_data_record["image_path"] = current_image_saved_path # 이미지 경로 업데이트 (없으면 None)
            

            metadata_f.write(json.dumps(sample_data_record, ensure_ascii=False) + "\n")
    print(f"\n모든 처리가 완료되었습니다. 결과는 '{ANALYSIS_OUTPUT_DIR}' 디렉토리를 확인하세요.")

if __name__ == "__main__":
    analyze_low_score_samples()