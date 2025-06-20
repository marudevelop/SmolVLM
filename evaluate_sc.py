import json
import re
import yaml
import numpy as np
from tqdm import tqdm

def _process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    articles = ['a', 'an', 'the']
    manualMap = {
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
    contractions = {
        'aint': "ain't",
        'arent': "aren't",
        'cant': "can't",
        'couldve': "could've",
        'couldnt': "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        'didnt': "didn't",
        'doesnt': "doesn't",
        'dont': "don't",
        'hadnt': "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        'hasnt': "hasn't",
        'havent': "haven't",
        'hed': "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        'hes': "he's",
        'howd': "how'd",
        'howll': "how'll",
        'hows': "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        'Im': "I'm",
        'Ive': "I've",
        'isnt': "isn't",
        'itd': "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        'itll': "it'll",
        "let's": "let's",
        'maam': "ma'am",
        'mightnt': "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        'mightve': "might've",
        'mustnt': "mustn't",
        'mustve': "must've",
        'neednt': "needn't",
        'notve': "not've",
        'oclock': "o'clock",
        'oughtnt': "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        'shant': "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        'shouldve': "should've",
        'shouldnt': "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": 'somebodyd',
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        'somebodyll': "somebody'll",
        'somebodys': "somebody's",
        'someoned': "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        'someonell': "someone'll",
        'someones': "someone's",
        'somethingd': "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        'somethingll': "something'll",
        'thats': "that's",
        'thered': "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        'therere': "there're",
        'theres': "there's",
        'theyd': "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        'theyll': "they'll",
        'theyre': "they're",
        'theyve': "they've",
        'twas': "'twas",
        'wasnt': "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        'weve': "we've",
        'werent': "weren't",
        'whatll': "what'll",
        'whatre': "what're",
        'whats': "what's",
        'whatve': "what've",
        'whens': "when's",
        'whered': "where'd",
        'wheres': "where's",
        'whereve': "where've",
        'whod': "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        'wholl': "who'll",
        'whos': "who's",
        'whove': "who've",
        'whyll': "why'll",
        'whyre': "why're",
        'whys': "why's",
        'wont': "won't",
        'wouldve': "would've",
        'wouldnt': "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        'yall': "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        'youd': "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        'youll': "you'll",
        'youre': "you're",
        'youve': "you've",
    }
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def process_punctuation(inText):
    outText = inText
    punct = [';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    commaStrip = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def process_answer(answer):
    answer = answer.replace('\n', ' ').replace('\t', ' ').strip()
    answer = process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer

def levenshtein_distance(s1, s2):
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

def anls_compute(groundtruth, prediction):
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(gt_answer), len(det_answer))
    value = 0.0 if length == 0 else float(dist) / float(length)
    return value

def evaluate_accuracy(prediction, ground_truths):
    """단일 예측에 대한 정확도를 계산"""
    processed_pred = process_answer(prediction)
    processed_gts = [process_answer(gt) for gt in ground_truths]
    return 1 if processed_pred in processed_gts else 0

def evaluate_anls(prediction, ground_truths):
    """단일 예측에 대한 ANLS 점수를 계산"""
    anls_per_gt = [anls_compute(gt, prediction) for gt in ground_truths]
    min_anls = min(anls_per_gt) if anls_per_gt else 1.0
    return 1.0 - min_anls

def main():
    results_files = [
        "results/SmolVLM_DocumentVQA_validation_seed42.json",
        # "results/ModelName_DocumentVQA_validation_seed42.json",
        # "results/SmolVLM_DocumentVQA_validation_seed42_corrected.json",
        "results/self-consistency_DocumentVQA_validation_seed42.json"
    ]

    for results_file in results_files:
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            print(f"Error: Results file not found at {results_file}")
            return

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        sample_size = config["dataset"]["sample_size"]

        if sample_size > 0 and sample_size < len(results):
            results = results[:sample_size]
        
        # 개별 답변들과 self-consistency 결과에 대한 정확도/ANLS 계산
        individual_accuracies = [[], [], []]  # 첫번째, 두번째, 세번째 답변
        individual_anls = [[], [], []]
        
        self_consistency_accuracies = []
        self_consistency_anls = []
        
        for item in tqdm(results, desc=f"Evaluating {results_file}"):
            ground_truths = item["answers"]
            
            # Self-consistency 결과 평가
            self_consistency_pred = item["response"]
            self_consistency_accuracies.append(evaluate_accuracy(self_consistency_pred, ground_truths))
            self_consistency_anls.append(evaluate_anls(self_consistency_pred, ground_truths))
            
            # 개별 답변들 평가 (voting_stats에서 original_responses 사용)
            if "voting_stats" in item and "original_responses" in item["voting_stats"]:
                original_responses = item["voting_stats"]["original_responses"]
                
                # 각 개별 답변에 대해 평가
                for i in range(min(3, len(original_responses))):  # 최대 3개까지
                    if i < len(original_responses):
                        individual_pred = original_responses[i]
                        individual_accuracies[i].append(evaluate_accuracy(individual_pred, ground_truths))
                        individual_anls[i].append(evaluate_anls(individual_pred, ground_truths))
        
        # 결과 출력
        print(f"\n=== Evaluation Results for {results_file} ===")
        print(f"Total samples processed: {len(results)}")
        
        # 개별 답변들의 결과
        for i in range(3):
            if individual_accuracies[i]:  # 해당 답변이 존재하는 경우만
                avg_acc = np.mean(individual_accuracies[i]) * 100
                avg_anls = np.mean(individual_anls[i])
                print(f"\n--- Individual Answer {i+1} ---")
                print(f"Samples: {len(individual_accuracies[i])}")
                print(f"Accuracy: {avg_acc:.2f}%")
                print(f"ANLS: {avg_anls:.4f}")
        
        # Self-consistency 결과
        sc_avg_accuracy = np.mean(self_consistency_accuracies) * 100
        sc_avg_anls = np.mean(self_consistency_anls)
        print(f"\n--- Self-Consistency ---")
        print(f"Samples: {len(self_consistency_accuracies)}")
        print(f"Accuracy: {sc_avg_accuracy:.2f}%")
        print(f"ANLS: {sc_avg_anls:.4f}")
        
        # 비교 요약
        print(f"\n--- Summary Comparison ---")
        for i in range(3):
            if individual_accuracies[i]:
                acc = np.mean(individual_accuracies[i]) * 100
                print(f"Answer {i+1} Accuracy: {acc:.2f}%")
        print(f"Self-Consistency Accuracy: {sc_avg_accuracy:.2f}%")
        print("=" * 50)

if __name__ == "__main__":
    main()