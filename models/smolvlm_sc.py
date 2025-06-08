import os.path as osp
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from PIL import Image
from collections import Counter
import re

def splitlen(s, sym='/'):
    return len(s.split(sym))

class SmolVLM_sc:
    def __init__(self, model_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct", device="cuda", **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        ).to(self.device)

        kwargs_default = {
            "max_new_tokens": 2048, 
            "do_sample": True,
            "temperature": 0.3,
            "use_cache": True
        }
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def extract_question_from_message(self, message):
        """메시지에서 질문 텍스트 추출"""
        for msg in message:
            if msg["type"] == "text":
                return msg["value"].strip()
        return ""

    def extract_images_from_message(self, message):
        """메시지에서 이미지들 추출"""
        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
        return images

    def generate_single_response(self, prompt, images):
        """단일 응답 생성"""
        images_processed = (
            [images] if isinstance(images, Image.Image) else images
        )

        inputs = self.processor(
            text=prompt, images=images_processed, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, **self.kwargs)

        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )[0]

        return generated_text.strip()

    def build_prompt_attention_guided_step1(self, message):
        """방법 1-1: 어디를 봐야 하는지 물어보기"""
        question = self.extract_question_from_message(message)
        images = self.extract_images_from_message(message)
        
        prompt = (
            f"<|im_start|>User:<image>Where in this document would I find the answer to: {question}?"
            f"<end_of_utterance>\nAssistant:"
        )
        
        return prompt, images

    def build_prompt_attention_guided_step2(self, message, focus_area):
        """방법 1-2: 해당 영역에 집중해서 답하기"""
        question = self.extract_question_from_message(message)
        images = self.extract_images_from_message(message)
        
        prompt = (
            f"<|im_start|>User:<image>Focus on {focus_area}. "
            f"Give a short and terse answer. No full stops. Question: {question}"
            f"<end_of_utterance>\nAssistant:"
        )
        
        return prompt, images

    def build_prompt_standard(self, message):
        """방법 2: Standard - 기존 방식"""
        prompt = (
            "<|im_start|>User:<image>Give a short and terse answer to the following question. "
            + "Do not paraphrase or reformat the text you see in the image. Do not include any full stops. "
            + "Just give the answer without additional explanation. Question: "
        )

        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_structure_first(self, message):
        """방법 3: Structure-First - 구조 파악 후 답하기"""
        prompt = (
            "<|im_start|>User:<image>Identify the document type (table/form/list), then give a short and terse answer. "
            + "Do not include full stops. Just give the answer. Question: "
        )

        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
            elif msg["type"] == "text":
                prompt += msg["value"].strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_model_choice(self, message, response_1, response_2, response_3):
        """모델한테 A/B/C 선택하게 하기"""
        question = self.extract_question_from_message(message)
        images = self.extract_images_from_message(message)
        
        prompt = (
            f"<|im_start|>User:<image>Question: {question}\n\n"
            f"Here are 3 possible answers:\n"
            f"A) {response_1}\n"
            f"B) {response_2}\n"
            f"C) {response_3}\n\n"
            f"Which answer is most accurate? Just reply A, B, or C."
            f"<end_of_utterance>\nAssistant:"
        )
        
        return prompt, images

    def normalize_answer(self, answer):
        """답변 정규화: 대소문자, 띄어쓰기, 구두점 제거"""
        if not answer:
            return ""
        
        # 소문자 변환
        normalized = answer.lower()
        
        # 구두점과 특수문자 제거 (알파벳, 숫자, 한글만 남기기)
        normalized = re.sub(r'[^\w\s가-힣]', '', normalized)
        
        # 연속된 공백을 하나로 줄이고 앞뒤 공백 제거
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def select_final_answer(self, message, response_1, response_2, response_3):
        """투표 시스템으로 최종 답변 선택 + 통계 정보 수집"""
        # 정규화된 답변들
        norm_1 = self.normalize_answer(response_1)
        norm_2 = self.normalize_answer(response_2) 
        norm_3 = self.normalize_answer(response_3)
        
        normalized_responses = [norm_1, norm_2, norm_3]
        original_responses = [response_1, response_2, response_3]
        
        # 투표 결과 분석
        vote_count = Counter(normalized_responses)
        unique_answers = len(vote_count)
        max_votes = vote_count.most_common(1)[0][1]
        
        # 통계 정보 저장
        voting_stats = {
            "original_responses": original_responses,
            "normalized_responses": normalized_responses,
            "unique_answers": unique_answers,
            "max_votes": max_votes,
            "vote_distribution": dict(vote_count)
        }
        
        print(f"Voting analysis: {unique_answers} unique answers, max votes: {max_votes}")
        
        if max_votes >= 2:  # 2표 이상 (2:1 또는 3:0)
            # 가장 많은 표를 받은 정규화된 답변 찾기
            winner_normalized = vote_count.most_common(1)[0][0]
            
            # 해당 정규화된 답변에 매칭되는 첫 번째 원본 답변 반환
            for i, norm_resp in enumerate(normalized_responses):
                if norm_resp == winner_normalized:
                    final_answer = original_responses[i]
                    break
            
            voting_stats["selection_method"] = "voting"
            print(f"Selected by voting: {final_answer}")
            
        else:  # 3개 모두 다름 (1:1:1)
            # 모델한테 선택하게 하기
            choice_prompt, images = self.build_prompt_model_choice(message, response_1, response_2, response_3)
            model_choice = self.generate_single_response(choice_prompt, images)
            
            print(f"Model choice: {model_choice}")
            
            # A, B, C 파싱
            if 'A' in model_choice.upper():
                final_answer = response_1
            elif 'B' in model_choice.upper():
                final_answer = response_2
            elif 'C' in model_choice.upper():
                final_answer = response_3
            else:
                print("Model choice unclear, defaulting to response_1")
                final_answer = response_1
            
            voting_stats["selection_method"] = "model_choice"
            voting_stats["model_choice_raw"] = model_choice
            print(f"Selected by model: {final_answer}")
        
        voting_stats["final_answer"] = final_answer
        return final_answer, voting_stats

    def generate_inner_with_stats(self, message, dataset=None):
        """Self-consistency + 통계 정보까지 반환하는 함수"""
        
        print("="*50)
        print("Starting Self-Consistency VQA")
        print("="*50)
        
        # 방법 1: Attention-Guided (2단계)
        print("\n[1/3] Attention-Guided Method:")
        step1_prompt, images = self.build_prompt_attention_guided_step1(message)
        focus_area = self.generate_single_response(step1_prompt, images)
        print(f"  Step 1 - Focus area: {focus_area}")
        
        step2_prompt, _ = self.build_prompt_attention_guided_step2(message, focus_area)
        response_1 = self.generate_single_response(step2_prompt, images)
        print(f"  Step 2 - Answer: {response_1}")
        
        # 방법 2: Standard
        print("\n[2/3] Standard Method:")
        standard_prompt, images = self.build_prompt_standard(message)
        response_2 = self.generate_single_response(standard_prompt, images)
        print(f"  Answer: {response_2}")
        
        # 방법 3: Structure-First
        print("\n[3/3] Structure-First Method:")
        structure_prompt, images = self.build_prompt_structure_first(message)
        response_3 = self.generate_single_response(structure_prompt, images)
        print(f"  Answer: {response_3}")
        
        # 최종 선택
        print("\n" + "="*30)
        print("FINAL SELECTION")
        print("="*30)
        final_answer, voting_stats = self.select_final_answer(message, response_1, response_2, response_3)
        print(f"Final Answer: {final_answer}")
        print("="*50)
        
        return final_answer, voting_stats

    def generate_inner(self, message, dataset=None):
        """기존 호환성을 위한 함수 (final_answer만 반환)"""
        final_answer, _ = self.generate_inner_with_stats(message, dataset)
        return final_answer

    # 기존 호환성을 위한 메서드
    def build_prompt_docvqa(self, message):
        return self.build_prompt_standard(message)