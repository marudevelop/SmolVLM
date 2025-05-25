import json
import re
# import string # 새로운 process_punctuation 함수는 string.punctuation을 사용하지 않으므로 주석 처리 또는 삭제 가능

# --- vqa_eval.py 및 제공된 smp_utils.py의 헬퍼 함수 및 상수 ---

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
    # _process_digit_article 함수는 내부적으로 입력을 소문자로 변환합니다.
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

# 사용자가 마지막으로 제공한 smp_utils.py의 process_punctuation 함수로 교체
def process_punctuation(inText: str) -> str: # 타입 힌트 추가
    # import re # re는 이미 파일 상단에 import 되어 있으므로 여기서 다시 import 할 필요 없음
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip  = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        # 원본 로직에서 inText를 참조하는 부분을 outText로 변경하여 누적된 변경사항에 대해 동작하도록 수정
        # (또는 원본 의도대로 inText를 계속 사용하는 것이 맞다면 그대로 두어야 합니다.
        #  여기서는 일반적인 텍스트 처리 흐름을 가정하여 outText로 변경했습니다.)
        if (p + ' ' in outText or ' ' + p in outText) or \
           (re.search(commaStrip, outText) is not None): # 원본은 inText를 참조
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def process_answer_for_anls(answer: str) -> str:
    """ANLS 계산을 위해 답변 텍스트를 처리합니다."""
    if answer is None:
        return ""
    answer = str(answer) # 문자열인지 확인
    answer = answer.replace('\n', ' ')
    answer = answer.replace('\t', ' ')
    answer = answer.strip()
    # 여기서 새로 업데이트된 process_punctuation 함수가 호출됩니다.
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
    # 입력 문자열은 process_answer_for_anls를 통해 이미 전처리되었음
    # 여기서 추가적인 .strip().lower().split()은 최종 정규화를 보장
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(gt_answer), len(det_answer))
    if length == 0:
        return 0.0 if dist == 0 else 1.0
    return float(dist) / float(length)

# --- regex_evaluate.py의 기존 함수 (필요시 수정) ---

def extract_answer_from_response(response: str) -> str:
    """
    'The answer is : ~~~' 형태에서 답변 부분을 추출합니다.
    """
    pattern = r'the answer is\s*:?\s*(.+?)(?:\n|$)'
    match = re.search(pattern, response.lower())
    
    if match:
        answer = match.group(1).strip()
        return answer
    
    return response.split('\n')[0].strip()

def is_correct_with_anls(predicted: str, ground_truths: list, anls_threshold: float = 0.5) -> tuple[bool, float]:
    """
    ANLS 점수를 기준으로 예측 답변이 정답인지 확인합니다.
    어떤 정답(ground truth)에 대해서든 ANLS 점수가 anls_threshold 이하이면 정답으로 간주합니다.
    반환값: (정답 여부, 최적 ANLS 점수) 튜플
    """
    processed_predicted = process_answer_for_anls(predicted)
    if not ground_truths:
        return False, 1.0

    min_anls_score = float('inf')
    
    for gt_raw in ground_truths: # 변수명 명확화
        processed_gt = process_answer_for_anls(str(gt_raw)) # gt도 문자열로 확실히 변환
        
        # 두 문자열이 모두 비어있는 경우 ANLS는 0 (동일)
        if not processed_gt and not processed_predicted:
             current_anls = 0.0
        # 한쪽만 비어있는 경우 ANLS는 1 (완전 불일치)
        elif (not processed_gt and processed_predicted) or \
             (processed_gt and not processed_predicted):
            current_anls = 1.0
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
    
    print(f"=== DocVQA ANLS 평가 결과 (임계값: {anls_threshold}) ===\n")
    
    for item in results:
        if model_name not in item.get("model_response", {}):
            continue
            
        total += 1
        
        question_id = item.get("question_id", "N/A") # ID가 없을 경우 대비
        question = item.get("question", "N/A")
        ground_truths = item.get("answers", [])
        # 모델 응답이 없는 경우를 대비하여 .get() 사용 및 기본값 제공
        model_response_data = item.get("model_response", {}).get(model_name, {})
        model_response_full = model_response_data.get("response", "")

        if not model_response_full: # 모델 응답이 비어있는 경우
            predicted_raw = ""
        else:
            predicted_raw = extract_answer_from_response(model_response_full)
        
        is_correct, best_anls = is_correct_with_anls(predicted_raw, ground_truths, anls_threshold)
        
        if is_correct:
            correct += 1
        #else:
            # print(f"❌ ID: {question_id}")
            # print(f"   질문: {question}")
            # print(f"   정답(Ground Truths): {ground_truths}")
            # print(f"   예측(Raw): '{predicted_raw}'")
            # print(f"   예측(처리 후): '{process_answer_for_anls(predicted_raw)}'") # 처리 후 예측값도 출력
            # print(f"   최적 ANLS 점수: {best_anls:.4f} (임계값: {anls_threshold})")
            # print(f"   전체 응답 (첫 100자): {model_response_full[:100]}...")
            # print()
            
    if total > 0:
        accuracy = correct / total
    else:
        accuracy = 0.0 # 부동소수점으로 초기화
        
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
    anls_eval_threshold = 0.5
    
    print(f"모델 평가 시작: {model_name_to_eval}, ANLS 임계값: {anls_eval_threshold}")
    evaluation_metrics = evaluate_docvqa(results_file, model_name_to_eval, anls_eval_threshold)
    
    if evaluation_metrics:
        print("\n평가 완료.")
        print(f"전체 정확도 (ANLS 임계값 {evaluation_metrics['anls_threshold']}): {evaluation_metrics['accuracy']:.4f}")