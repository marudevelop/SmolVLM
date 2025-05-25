import json
import re
import string # 구두점 처리를 위해 추가

# --- vqa_eval.py의 헬퍼 함수 및 상수 ---
# 제공된 vqa_eval.py 스니펫에서 추출 및 적용

MANUAL_MAP = {
    'none': '0',
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
}

CONTRACTIONS = {
    'aint': "ain't", 'arent': "aren't", 'cant': "can't", 'couldve': "could've",
    'couldnt': "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    'didnt': "didn't", 'doesnt': "doesn't", 'dont': "don't", 'hadnt': "hadn't",
    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", 'hasnt': "hasn't",
    'havent': "haven't", 'hed': "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    'hes': "he's", 'howd': "how'd", 'howll': "how'll", 'hows': "how's",
    "Id've": "I'd've", "I'dve": "I'd've", 'Im': "I'm", 'Ive': "I've",
    'isnt': "isn't", 'itd': "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    'itll': "it'll", "let's": "let's", 'maam': "ma'am", 'mightnt': "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", 'mightve': "might've",
    'mustnt': "mustn't", 'mustve': "must've", 'neednt': "needn't",
    'notve': "not've", 'oclock': "o'clock", 'oughtnt': "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    'shant': "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", 'shouldve': "should've", 'shouldnt': "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": 'somebodyd', "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", 'somebodyll': "somebody'll",
    'somebodys': "somebody's", 'someoned': "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    'someonell': "someone'll", 'someones': "someone's",
    'somethingd': "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", 'somethingll': "something'll",
    'thats': "that's", 'thered': "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", 'therere': "there're", 'theres': "there's",
    'theyd': "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    'theyll': "they'll", 'theyre': "they're", 'theyve': "they've",
    'twas': "'twas", 'wasnt': "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", 'weve': "we've", 'werent': "weren't",
    'whatll': "what'll", 'whatre': "what're", 'whats': "what's",
    'whatve': "what've", 'whens': "when's", 'whered': "where'd",
    'wheres': "where's", 'whereve': "where've", 'whod': "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", 'wholl': "who'll",
    'whos': "who's", 'whove': "who've", 'whyll': "why'll",
    'whyre': "why're", 'whys': "why's", 'wont': "won't",
    'wouldve': "would've", 'wouldnt': "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    'yall': "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", 'youd': "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", 'youll': "you'll", 'youre': "you're",
    'youve': "you've"
}

def _process_digit_article(in_text: str) -> str:
    """숫자와 관사를 처리합니다."""
    out_text = []
    temp_text = in_text.lower().split()
    articles = ['a', 'an', 'the']
    for word in temp_text:
        word = MANUAL_MAP.setdefault(word, word)
        if word not in articles:
            out_text.append(word)
    for word_id, word in enumerate(out_text):
        if word in CONTRACTIONS:
            out_text[word_id] = CONTRACTIONS[word]
    return ' '.join(out_text)

def process_punctuation(in_text: str) -> str:
    """구두점을 처리합니다. VQA/DocVQA 작업에서 일반적인 방식입니다.
    소문자로 변환하고, 구두점을 공백으로 바꾼 후, 공백을 정규화합니다."""
    text = in_text.lower()
    for punc_char in string.punctuation:
        text = text.replace(punc_char, ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_answer_for_anls(answer: str) -> str:
    """ANLS 계산을 위해 답변 텍스트를 처리합니다."""
    if answer is None:
        return ""
    answer = str(answer) # 문자열인지 확인
    answer = answer.replace('\n', ' ')
    answer = answer.replace('\t', ' ')
    answer = answer.strip()
    answer = process_punctuation(answer)
    answer = _process_digit_article(answer)
    # anls_compute 함수 자체에서 최종적으로 lower() 및 strip()을 수행합니다.
    return answer

def levenshtein_distance(s1: str, s2: str) -> int:
    """두 문자열 간의 레벤슈타인 거리를 계산합니다."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth: str, prediction: str) -> float:
    """정규화된 레벤슈타인 거리(ANLS의 기반)를 계산합니다."""
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(gt_answer), len(det_answer)) # (이전에는 groundtruth.upper() 사용)에서 수정됨
    if length == 0: # 두 문자열 모두 비어있는 경우
        return 0.0 if dist == 0 else 1.0 # dist도 0이면 완벽 일치, 아니면 불일치
    return float(dist) / float(length)

# --- regex_evaluate.py의 기존 함수 (필요시 수정) ---

def extract_answer_from_response(response: str) -> str:
    """
    'The answer is : ~~~' 형태에서 답변 부분을 추출합니다.
    """
    # 대소문자 구분 없이 "the answer is" 패턴 찾기
    pattern = r'the answer is\s*:?\s*(.+?)(?:\n|$)'
    match = re.search(pattern, response.lower())
    
    if match:
        answer = match.group(1).strip()
        return answer
    
    # 패턴이 없으면 전체 응답의 첫 번째 줄 반환
    return response.split('\n')[0].strip()

def is_correct_with_anls(predicted: str, ground_truths: list, anls_threshold: float = 0.5) -> tuple[bool, float]:
    """
    ANLS 점수를 기준으로 예측 답변이 정답인지 확인합니다.
    어떤 정답(ground truth)에 대해서든 ANLS 점수가 anls_threshold 이하이면 정답으로 간주합니다.
    반환값: (정답 여부, 최적 ANLS 점수) 튜플
    """
    processed_predicted = process_answer_for_anls(predicted)
    if not ground_truths: # 비교할 정답이 없는 경우
        return False, 1.0 # 오답, 최대 ANLS 점수

    min_anls_score = float('inf')
    
    for gt in ground_truths:
        processed_gt = process_answer_for_anls(gt)
        if not processed_gt and not processed_predicted: # 처리 후 둘 다 비어있는 경우
             current_anls = 0.0
        elif not processed_gt or not processed_predicted: # 하나는 비어있고 다른 하나는 아닌 경우
            current_anls = 1.0 # 하나는 비어있고 다른 하나는 비어있지 않으면 최대 거리 (둘 다 비어있지 않은 한)
        else:
            current_anls = anls_compute(processed_gt, processed_predicted)
        
        if current_anls < min_anls_score:
            min_anls_score = current_anls
            
    return min_anls_score <= anls_threshold, min_anls_score

def evaluate_docvqa(results_file: str, model_name: str = "SmolVLM", anls_threshold: float = 0.5):
    """
    ANLS를 사용하여 DocVQA 결과를 평가합니다.
    """
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"오류: 결과 파일 '{results_file}'을(를) 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError:
        print(f"오류: '{results_file}'에서 JSON을 디코딩할 수 없습니다.")
        return None

    total = 0
    correct = 0
    
    print(f"=== DocVQA ANLS 평가 결과 (임계값: {anls_threshold}) ===\n") # 수정됨
    
    for item in results:
        if model_name not in item.get("model_response", {}):
            continue
            
        total += 1
        
        question_id = item["question_id"]
        question = item["question"]
        ground_truths = item["answers"]
        model_response_full = item["model_response"][model_name]["response"]
        
        predicted_raw = extract_answer_from_response(model_response_full)
        
        is_correct, best_anls = is_correct_with_anls(predicted_raw, ground_truths, anls_threshold) # 수정됨
        
        if is_correct:
            correct += 1
        #else:
            # 오답 상세 정보 출력
            # print(f"❌ ID: {question_id}")
            # print(f"   질문: {question}")
            # print(f"   정답(Ground Truths): {ground_truths}")
            # print(f"   예측(Raw): '{predicted_raw}'")
            # print(f"   예측(처리 후): '{process_answer_for_anls(predicted_raw)}'")
            # print(f"   최적 ANLS 점수: {best_anls:.4f} (임계값: {anls_threshold})") # 수정됨
            # print(f"   전체 응답: {model_response_full[:100]}...")
            # print()
            
    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0
        
    print(f"📊 최종 결과:")
    print(f"   총 질문 수: {total}")
    print(f"   정답 수 (ANLS <= {anls_threshold}): {correct}") 
    print(f"   정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "anls_threshold": anls_threshold
    }

if __name__ == "__main__":
    results_file = "results/docvqa_results.json"  
    model_name_to_eval = "SmolVLM"  
    anls_eval_threshold = 0.5 # DocVQA에 대한 표준 ANLS 임계값
    
    print(f"모델 평가 시작: {model_name_to_eval}, ANLS 임계값: {anls_eval_threshold}")
    evaluation_metrics = evaluate_docvqa(results_file, model_name_to_eval, anls_eval_threshold)
    
    if evaluation_metrics:
        print("\n평가 완료.")
        print(f"전체 정확도 ({evaluation_metrics['anls_threshold']}): {evaluation_metrics['accuracy']:.4f}")