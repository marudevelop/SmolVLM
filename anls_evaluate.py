import json
import re
import numpy as np
import Levenshtein

# --- Document 1과 2의 표준 방식을 따른 VQA 정규화 클래스 ---

class VQANormalizationGtVisionLab:
    def __init__(self):
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manual_map = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.comma_strip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def processPunctuation(self, in_text):
        out_text = in_text
        for p in self.punct:
            if (p + " " in in_text or " " + p in in_text) or (re.search(self.comma_strip, in_text) is not None):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        out_text = self.period_strip.sub("", out_text, re.UNICODE)
        return out_text

    def processDigitArticle(self, in_text):
        out_text = []
        tempText = in_text.lower().split()
        for word in tempText:
            word = self.manual_map.setdefault(word, word)
            if word not in self.articles:
                out_text.append(word)
            else:
                pass
        for wordId, word in enumerate(out_text):
            if word in self.contractions:
                out_text[wordId] = self.contractions[word]
        out_text = " ".join(out_text)
        return out_text

    def vqa_normalize_text(self, text):
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.strip()

        text = self.processPunctuation(text)
        text = self.processDigitArticle(text)
        return text


# --- Document 1의 표준 ANLS 계산 함수들 ---

def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0


def average_normalized_levenshtein_similarity(ground_truth, predicted_answers):
    assert len(ground_truth) == len(predicted_answers), "Length of ground_truth and predicted_answers must match."

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = ground_truth[i]
        o_q_i = predicted_answers[i]
        if o_q_i == "":
            max_score = 0
        else:
            max_score = max(similarity_score(a_ij, o_q_i) for a_ij in a_i)

        total_score += max_score

    return total_score / N


def evaluate_docvqa(results_file: str, model_name: str = "SmolVLM", similarity_threshold: float = 0.5):
    """
    DocVQA 결과를 표준 ANLS 방식으로 평가합니다.
    Document 1과 2의 표준 방식을 따릅니다.
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

    normalizer = VQANormalizationGtVisionLab()
    total_items = 0
    all_item_scores = []
    
    print(f"=== DocVQA ANLS 평가 결과 (표준 방식, 유사도 임계값: {similarity_threshold}) ===\n")
    
    for item in results:
        if model_name not in item.get("model_response", {}):
            continue
            
        total_items += 1
        
        question_id = item.get("question_id", "N/A")
        question = item.get("question", "N/A")
        ground_truths = item.get("answers", [])
        model_response_data = item.get("model_response", {}).get(model_name, {})
        model_response_full = model_response_data.get("response", "")

        # 전체 응답을 그대로 사용하고 표준 정규화 적용
        predicted_raw = model_response_full if model_response_full else ""
        processed_predicted = normalizer.vqa_normalize_text(predicted_raw)
        
        # 각 ground truth에 대해 표준 정규화 적용
        processed_ground_truths = []
        for gt_raw in ground_truths:
            processed_gt = normalizer.vqa_normalize_text(str(gt_raw))
            processed_ground_truths.append(processed_gt)
        
        # 표준 ANLS 방식으로 점수 계산
        if not processed_ground_truths:
            item_score = 0.0
        elif processed_predicted == "":
            item_score = 0.0
        else:
            # 각 ground truth와의 similarity_score 계산 후 최대값 선택
            max_score = max(similarity_score(gt, processed_predicted, similarity_threshold) 
                          for gt in processed_ground_truths)
            item_score = max_score
            
        all_item_scores.append(item_score)

        # 낮은 점수 항목 출력 (디버깅용)
        # if item_score < similarity_threshold:
        #     print(f"⚠️ ID: {question_id} (낮은 점수)")
        #     print(f"   질문: {question}")
        #     print(f"   정답(Ground Truths): {ground_truths}")
        #     print(f"   예측(전체 응답): '{predicted_raw[:200]}{'...' if len(predicted_raw) > 200 else ''}'")
        #     print(f"   처리된 예측: '{processed_predicted}'")
        #     print(f"   처리된 정답: {processed_ground_truths}")
        #     print(f"   항목 점수: {item_score:.4f}")
        #     print()
            
    overall_score = 0.0
    if total_items > 0:
        overall_score = np.mean(all_item_scores) * 100 
        
    print(f"\n📊 최종 결과:")
    print(f"   총 처리된 질문 수: {total_items}")
    print(f"   전체 점수 (표준 ANLS 방식): {overall_score:.2f}") 
    
    return {
        "total_questions": total_items,
        "overall_score": overall_score,
        "similarity_threshold": similarity_threshold 
    }


if __name__ == "__main__":
    results_file = "results/docvqa_results_exact2prompt.json"  
    model_name_to_eval = "SmolVLM"                

    similarity_eval_threshold = 0.5               
    
    print(f"모델 평가 시작: {model_name_to_eval}, 유사도 임계값: {similarity_eval_threshold}")
    evaluation_metrics = evaluate_docvqa(results_file, model_name_to_eval, similarity_eval_threshold)
    
    if evaluation_metrics:
        print("\n평가 완료.")
        print(f"전체 점수 (유사도 임계값 {evaluation_metrics['similarity_threshold']}): {evaluation_metrics['overall_score']:.2f}")