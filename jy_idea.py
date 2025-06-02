import os
import json
import time
import torch
import yaml
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from datasets import load_dataset
from models.smolvlm import SmolVLM

def get_few_shot_examples():
    """Few-shot examples for step-by-step reasoning with bbox annotations"""
    examples = [
        {
            "question": "How many people are in the image?",
            "reasoning": """Step 1: I need to identify all human figures in the image.
bbox: [50, 100, 150, 300], [200, 80, 280, 320], [350, 120, 420, 310]

Step 2: I can see three distinct people - one person on the left, one in the center, and one on the right.
bbox: [50, 100, 150, 300], [200, 80, 280, 320], [350, 120, 420, 310]

Step 3: Counting the identified individuals to get the final answer.

So, the answer is "3"."""
        },
        {
            "question": "What color is the car in the parking lot?",
            "reasoning": """Step 1: I need to locate the parking lot area in the image.
bbox: [100, 200, 400, 350]

Step 2: Within the parking lot, I can identify a car.
bbox: [180, 220, 280, 300]

Step 3: I can observe that the car appears to be red in color.
bbox: [180, 220, 280, 300]

So, the answer is "red"."""
        },
        {
            "question": "Is the person wearing a hat?",
            "reasoning": """Step 1: I need to locate the person in the image.
bbox: [120, 50, 220, 300]

Step 2: I need to focus on the head area to check for any headwear.
bbox: [130, 50, 210, 120]

Step 3: I can see that there is a hat on the person's head.
bbox: [135, 55, 205, 95]

So, the answer is "yes"."""
        }
    ]
    return examples

def create_first_stage_prompt(question, examples):
    """Create prompt for first stage: step-by-step reasoning with bbox"""
    prompt = "You are an expert at visual question answering. Please analyze the image step by step and provide bounding boxes for each reasoning step.\n\n"
    
    # Add few-shot examples
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Question: {example['question']}\n"
        prompt += f"Analysis:\n{example['reasoning']}\n\n"
    
    prompt += f"Now, please analyze this image:\nQuestion: {question}\n"
    prompt += "Analysis:\n"
    prompt += "Please provide step-by-step reasoning with bounding boxes in the format [x1, y1, x2, y2] for each step, and end with 'So, the answer is \"[your answer]\"'."
    
    return prompt

def extract_bboxes_from_response(response):
    """First stage response에서 bbox 좌표들을 순서대로 추출 (중복 허용)"""
    # bbox 패턴 매칭
    bbox_pattern = r'bbox:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    
    bboxes_with_steps = []
    lines = response.split('\n')
    
    step_counter = 0
    for line in lines:
        # Step 감지
        if line.strip().startswith('Step'):
            step_counter += 1
        
        # bbox 찾기
        matches = re.findall(bbox_pattern, line)
        for match in matches:
            x1, y1, x2, y2 = map(int, match)
            # 유효한 bbox인지 체크
            if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                bboxes_with_steps.append({
                    'bbox': [x1, y1, x2, y2],
                    'step': step_counter,
                    'line': line.strip()
                })
    
    print(f"상세 bbox 추출 결과:")
    for i, item in enumerate(bboxes_with_steps):
        print(f"  {i+1}. Step {item['step']}: {item['bbox']} - {item['line'][:50]}...")
    
    # bbox 좌표만 반환 (순서 유지, 중복 포함)
    return [item['bbox'] for item in bboxes_with_steps]

def draw_bboxes_on_image(image, bboxes):
    """이미지에 bbox를 그려서 시각화"""
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image.copy()
    
    # 색상 팔레트
    colors = [
        (255, 0, 0),    # 빨강
        (0, 255, 0),    # 초록  
        (0, 0, 255),    # 파랑
        (255, 255, 0),  # 노랑
        (255, 0, 255),  # 자홍
        (0, 255, 255),  # 청록
        (255, 128, 0),  # 주황
        (128, 0, 255),  # 보라
    ]
    
    print(f"이미지에 {len(bboxes)}개의 bbox 그리기")
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        color = colors[i % len(colors)]
        
        # bbox 그리기 (두껍게)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 4)
        
        # Step 번호 라벨 추가
        label = f"Step {i+1}"
        
        # 라벨 배경 그리기
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(image_np, (x1, y1-text_height-10), (x1+text_width+10, y1), color, -1)
        
        # 라벨 텍스트 (흰색)
        cv2.putText(image_np, label, (x1+5, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return Image.fromarray(image_np)

def create_second_stage_prompt(question, first_response, bbox_count):
    """Stage 2 프롬프트: bbox가 시각화된 이미지를 보고 검증 후 최종 답변"""
    
    prompt = f"""You are an expert reviewer for visual question answering. The image shows colored bounding boxes with step labels that correspond to the reasoning steps from the first analysis.

Original Question: {question}

First Stage Analysis:
{first_response}

The image now has {bbox_count} colored bounding boxes drawn on it, each labeled with "Step X" numbers. Each box corresponds to the reasoning steps above.

Your task:
1. **Verify Box Accuracy**: Check if each colored box accurately highlights the areas mentioned in the corresponding reasoning step
2. **Assess Spatial Logic**: Determine if the bounding box locations are appropriate for the analysis
3. **Evaluate Step Consistency**: Confirm that each step's bbox matches what is described in the reasoning
4. **Review Overall Logic**: Assess if the step-by-step reasoning is sound and well-supported

After your verification, you must provide a definitive final answer to the original question.

Please examine each colored box carefully and provide your assessment followed by your final answer.

Final Answer: [your definitive answer to the question]"""

    return prompt

def extract_final_answer(response):
    """Stage 2 응답에서 최종 답변 추출"""
    # "Final Answer:" 패턴 찾기
    patterns = [
        r'Final Answer:\s*"([^"]+)"',
        r'Final Answer:\s*([^\n]+)',
        r'final answer is\s*"([^"]+)"',
        r'final answer is\s*([^\n]+)',
        r'answer is\s*"([^"]+)"',
        r'answer is\s*([^\n]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # 패턴을 찾지 못하면 응답의 마지막 줄 반환
    lines = response.strip().split('\n')
    if lines:
        return lines[-1].strip()
    
    return "No answer found"

def enhanced_two_stage_inference(model, image, question, few_shot_examples):
    """개선된 2단계 추론 - bbox 시각화 활용"""
    
    # Stage 1: Step-by-step reasoning with bbox
    first_stage_prompt = create_first_stage_prompt(question, few_shot_examples)
    message_stage1 = [
        {"type": "image", "value": image},
        {"type": "text", "value": first_stage_prompt}
    ]
    
    print("Stage 1: 단계별 분석 시작...")
    start_time_stage1 = time.time()
    first_response = model.generate_inner(message_stage1)
    end_time_stage1 = time.time()
    stage1_time = end_time_stage1 - start_time_stage1
    print(f"Stage 1 완료 (소요시간: {stage1_time:.2f}초)")
    
    # Stage 1에서 bbox 추출
    extracted_bboxes = extract_bboxes_from_response(first_response)
    print(f"추출된 bbox 개수: {len(extracted_bboxes)}")
    
    # Stage 2: Verification with bbox visualization
    start_time_stage2 = time.time()
    
    if extracted_bboxes:
        try:
            print("Stage 2: bbox 시각화 및 검증 시작...")
            # 이미지에 bbox 시각화
            annotated_image = draw_bboxes_on_image(image, extracted_bboxes)
            
            # Stage 2 프롬프트 생성
            stage2_prompt = create_second_stage_prompt(question, first_response, len(extracted_bboxes))
            
            message_stage2 = [
                {"type": "image", "value": annotated_image},
                {"type": "text", "value": stage2_prompt}
            ]
            
            # Stage 2 실행
            second_response = model.generate_inner(message_stage2)
            
            # 최종 답변 추출
            final_answer = extract_final_answer(second_response)
            
            used_visualization = True
            print(f"Stage 2 완료 (bbox 시각화 사용)")
            
        except Exception as e:
            print(f"bbox 시각화 실패: {e}")
            # 폴백: 원본 이미지로 간단 검증
            second_response = simple_verification(model, image, question, first_response)
            final_answer = extract_final_answer(second_response)
            used_visualization = False
    else:
        print("bbox 없음 - 간단 검증 모드")
        # bbox가 없으면 간단 검증
        second_response = simple_verification(model, image, question, first_response)
        final_answer = extract_final_answer(second_response)
        used_visualization = False
    
    end_time_stage2 = time.time()
    stage2_time = end_time_stage2 - start_time_stage2
    print(f"Stage 2 완료 (소요시간: {stage2_time:.2f}초)")
    
    return {
        'first_response': first_response,
        'second_response': second_response,
        'final_answer': final_answer,
        'stage1_time': stage1_time,
        'stage2_time': stage2_time,
        'used_visualization': used_visualization,
        'num_bboxes': len(extracted_bboxes)
    }

def simple_verification(model, image, question, first_response):
    """bbox가 없거나 시각화가 실패했을 때의 간단 검증"""
    
    prompt = f"""Please review the following visual analysis and provide a final answer.

Original Question: {question}

Analysis to Review:
{first_response}

Looking at the original image, please verify if the analysis is correct and provide your final answer.

Final Answer: [your answer]"""

    message = [
        {"type": "image", "value": image},
        {"type": "text", "value": prompt}
    ]
    
    return model.generate_inner(message)

def main():
    # 설정 로드
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_classes = {
        "SmolVLM": SmolVLM,
    }
    
    # 데이터셋 설정
    dataset_path = "HuggingFaceM4/DocumentVQA"
    dataset_name = dataset_path.split("/")[-1]
    dataset_split = "validation"
    
    sample_size = config["dataset"]["sample_size"]
    sample_seed = config["dataset"]["sample_seed"]
    device = config["device"]
    
    # 데이터셋 로드
    dataset = load_dataset(dataset_path, split=dataset_split)
    if sample_size > 0:
        dataset = dataset.shuffle(seed=sample_seed).select(range(sample_size))
    
    # 결과 디렉토리 생성
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Few-shot examples
    few_shot_examples = get_few_shot_examples()
    
    # 각 모델에 대해 실행
    for model_name, ModelClass in model_classes.items():
        output_filename = f"results/{model_name}_{dataset_name}_{dataset_split}_bbox_visualization_seed{sample_seed}.json"
        
        # 기존 결과 로드
        results = []
        processed_qids = set()
        
        if os.path.exists(output_filename):
            with open(output_filename, "r") as f:
                try:
                    results = json.load(f)
                    processed_qids = {item["question_id"] for item in results}
                    print(f"기존 결과 {len(results)}개 로드됨 (모델: {model_name})")
                except json.JSONDecodeError:
                    print(f"경고: {output_filename} 파일을 읽을 수 없습니다. 새로 시작합니다.")
        
        print(f"\n모델 {model_name}에 대한 2단계 bbox 시각화 추론 시작")
        model = ModelClass(device=device)
        
        # 데이터셋 처리
        for item in tqdm(dataset, desc=f"Processing {model_name}"):
            question_id = item["questionId"]
            
            # 이미 처리된 항목은 건너뛰기
            if question_id in processed_qids:
                continue
            
            image = item["image"]
            question = item["question"]
            answers = item["answers"]
            
            try:
                print(f"\n처리 중: {question_id}")
                print(f"질문: {question}")
                
                # 2단계 추론 실행
                result = enhanced_two_stage_inference(model, image, question, few_shot_examples)
                
                # 결과 저장
                result_item = {
                    "question_id": question_id,
                    "question": question,
                    "ground_truth_answers": answers,
                    "stage1_response": result['first_response'],
                    "stage2_response": result['second_response'],
                    "final_answer": result['final_answer'],
                    "stage1_inference_time": result['stage1_time'],
                    "stage2_inference_time": result['stage2_time'],
                    "total_inference_time": result['stage1_time'] + result['stage2_time'],
                    "used_bbox_visualization": result['used_visualization'],
                    "num_extracted_bboxes": result['num_bboxes']
                }
                
                results.append(result_item)
                processed_qids.add(question_id)
                
                print(f"최종 답변: {result['final_answer']}")
                print(f"bbox 시각화 사용: {result['used_visualization']}")
                
                # 매번 저장 (데이터 손실 방지)
                with open(output_filename, "w") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                    
            except Exception as e:
                print(f"오류 발생 (question_id: {question_id}): {e}")
                continue
        
        # 최종 통계
        print(f"\n{model_name} 처리 완료!")
        print(f"결과 파일: {output_filename}")
        
        if results:
            bbox_viz_used = sum(1 for r in results if r.get("used_bbox_visualization", False))
            avg_bboxes = sum(r.get("num_extracted_bboxes", 0) for r in results) / len(results)
            avg_time = sum(r.get("total_inference_time", 0) for r in results) / len(results)
            
            print(f"\n통계:")
            print(f"- 총 처리된 항목: {len(results)}")
            print(f"- bbox 시각화 사용: {bbox_viz_used}/{len(results)} ({bbox_viz_used/len(results)*100:.1f}%)")
            print(f"- 평균 bbox 개수: {avg_bboxes:.2f}")
            print(f"- 평균 처리 시간: {avg_time:.2f}초")

if __name__ == "__main__":
    main()