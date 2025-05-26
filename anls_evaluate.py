import json
import re
import numpy as np 

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
    """숫자와 관사를 처리합니다. (내부적으로 소문자 변환 포함)"""
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

def process_punctuation(inText: str) -> str:
    """제공된 smp_utils.py의 구두점 처리 로직."""
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip  = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        if (p + ' ' in outText or ' ' + p in outText) or \
           (re.search(commaStrip, outText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def process_answer_for_anls(answer: str) -> str:
    """ANLS 계산을 위해 답변 텍스트를 전처리합니다."""
    if answer is None:
        return ""
    answer = str(answer)
    answer = answer.replace('\n', ' ').replace('\t', ' ')
    answer = answer.strip()
    answer = process_punctuation(answer)
    answer = _process_digit_article(answer)
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
    length = max(len(gt_answer), len(det_answer))
    if length == 0:
        return 0.0 if dist == 0 else 1.0
    return float(dist) / float(length)

def evaluate_docvqa(results_file: str, model_name: str = "SmolVLM", similarity_threshold: float = 0.5):
    """
    DocVQA 결과를 VLMEvalKit의 ANLS 점수 계산 방식과 유사하게 평가합니다.
    전체 응답을 그대로 사용하여 평가합니다.
    similarity_threshold: 1 - ANLS_distance 에 대한 임계값 
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

    total_items = 0
    all_item_scores = []
    
    print(f"=== DocVQA ANLS 평가 결과 (전체 응답 기준, 유사도 임계값: {similarity_threshold}) ===\n")
    
    for item in results:
        if model_name not in item.get("model_response", {}):
            continue
            
        total_items += 1
        
        question_id = item.get("question_id", "N/A")
        question = item.get("question", "N/A")
        ground_truths = item.get("answers", [])
        model_response_data = item.get("model_response", {}).get(model_name, {})
        model_response_full = model_response_data.get("response", "")

        # 전체 응답을 그대로 사용 (정규식으로 자르지 않음)
        predicted_raw = model_response_full if model_response_full else ""
        
        processed_predicted = process_answer_for_anls(predicted_raw)
        
        min_anls_dist_for_item = 1.0  
        if ground_truths:
            current_item_anls_distances = []
            for gt_raw in ground_truths:
                processed_gt = process_answer_for_anls(str(gt_raw))
                if not processed_gt and not processed_predicted:
                    anls_dist = 0.0
                elif (not processed_gt and processed_predicted) or \
                     (processed_gt and not processed_predicted):
                    anls_dist = 1.0
                else:
                    anls_dist = anls_compute(processed_gt, processed_predicted)
                current_item_anls_distances.append(anls_dist)
            
            if current_item_anls_distances: # ANLS 거리가 하나라도 계산된 경우
                min_anls_dist_for_item = min(current_item_anls_distances)
                
        
        similarity = 1.0 - min_anls_dist_for_item
        item_score = 0.0 if similarity < similarity_threshold else similarity
        all_item_scores.append(item_score)

        # if item_score < similarity_threshold:
        #     print(f"⚠️ ID: {question_id} (낮은 점수)")
        #     print(f"   질문: {question}")
        #     print(f"   정답(Ground Truths): {ground_truths}")
        #     print(f"   예측(전체 응답): '{predicted_raw[:200]}{'...' if len(predicted_raw) > 200 else ''}'")
        #     print(f"   최소 ANLS 거리: {min_anls_dist_for_item:.4f}")
        #     print(f"   항목 점수: {item_score:.4f} (유사도: {similarity:.4f})")
        #     print()
            
    overall_score = 0.0
    if total_items > 0:
        overall_score = np.mean(all_item_scores) * 100 
        
    print(f"\n📊 최종 결과:")
    print(f"   총 처리된 질문 수: {total_items}")
    print(f"   전체 점수 (VLMEvalKit DocVQA 방식, 전체 응답 기준): {overall_score:.2f}") 
    
    return {
        "total_questions": total_items,
        "overall_score": overall_score,
        "similarity_threshold": similarity_threshold 
    }

if __name__ == "__main__":
    results_file = "results/docvqa_results_exact2paper.json"  
    model_name_to_eval = "SmolVLM"                

    similarity_eval_threshold = 0.5               
    
    print(f"모델 평가 시작: {model_name_to_eval}, 유사도 임계값: {similarity_eval_threshold}")
    evaluation_metrics = evaluate_docvqa(results_file, model_name_to_eval, similarity_eval_threshold)
    
    if evaluation_metrics:
        print("\n평가 완료.")
        print(f"전체 점수 (유사도 임계값 {evaluation_metrics['similarity_threshold']}): {evaluation_metrics['overall_score']:.2f}")