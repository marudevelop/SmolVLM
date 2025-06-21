import os.path as osp
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from PIL import Image

def splitlen(s, sym='/'):
    return len(s.split(sym))

class SmolVLM:
    def __init__(self, model_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct", device="cuda", **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.device = device
        self.sampling_frames = 64
        self.resolution = 384

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        ).to(self.device)

        kwargs_default = {"max_new_tokens": 2048, "do_sample": False, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _generate_single_response(self, prompt, images=None, temperature=0.0, **kwargs): # **kwargs 추가
        # 이미지 유무에 따라 inputs 생성
        if images:
            images_processed = (
                [images]
                if isinstance(images, Image.Image)
                else images
            )
            inputs = self.processor(
                text=prompt, images=images_processed, return_tensors="pt"
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=prompt, return_tensors="pt"
            ).to(self.model.device)

        gen_kwargs = self.kwargs.copy()
        if temperature > 0:
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = temperature
        else:
            gen_kwargs['do_sample'] = False
        
        gen_kwargs.update(kwargs) # 전달된 kwargs를 gen_kwargs에 업데이트

        generated_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )[0]

        return generated_text.strip()


    def _extract_keywords(self, question):
        """
        [Stage 1: 키워드 추출]
        질문 텍스트만 사용하여 핵심 키워드를 추출합니다.
        """

        print("  [Stage 1] Extracting keywords from question...")

        prompt = (
        f"<|im_start|>User:Extract the most relevant keywords from the following question. "
        f"The keywords you extract should be single words or short phrases. "
        f"Provide only the comma-separated keywords as the answer. Do not include any other text or explanation.\n\n"

        # Example 1: Basic object identification
        f"Question: What is the model name of the car in the image?\n"
        f"Keywords: car, model name\n\n"
        
        # Example 2 
        f"Question: How is the patient's current condition described in the medical report?\n"
        f"Keywords: patient's current condition, medical report\n\n"

        # Example 3: Extracting specific information from a document
        f"Question: Please find the grand total on this invoice.\n"
        f"Keywords: invoice, grand total\n\n"
        
        # Example 4: Identifying a role or person
        f"Question: Who is the author of the book cover shown in the image?\n"
        f"Keywords: author, book cover\n\n"

        # Example 5: Analyzing a chart or graph
        f"Question: According to the bar chart, which country had the highest GDP in 2023?\n"
        f"Keywords: bar chart, highest GDP, country, 2023\n\n"
        
        f"Question: {question} <end_of_utterance>\nAssistant:"
    )

        raw_keywords = self._generate_single_response(
        prompt, 
        images=None, 
        temperature=0.0,
        max_new_tokens=50,
        no_repeat_ngram_size=2,
        early_stopping=True
        )
        print(f"  [Stage 1] Raw output from model: {raw_keywords}")

        # --- 후처리 로직 시작 ---
        # 모델이 생성한 텍스트를 소문자로 변경하여 "keywords:"로 시작하는지 확인
        cleaned_keywords = raw_keywords
        if cleaned_keywords.lower().startswith('keywords:'):
            # "Keywords: " 부분을 잘라내고 앞뒤 공백 제거
            cleaned_keywords = cleaned_keywords[len('keywords:'):].strip()
        
        # 가끔 모델이 답변 양쪽에 추가하는 따옴표도 제거
        cleaned_keywords = cleaned_keywords.strip('\"\'')
        # --- 후처리 로직 끝 ---

        print(f"  [Stage 1] Cleaned keywords: {cleaned_keywords}")
        return cleaned_keywords

    def _answer_with_keywords_context(self, image, keywords, question):
        """
        [Stage 2: 이미지와 키워드를 기반으로 QA 답변 생성]
        """
        print("  [Stage 2] Answering question using image with extracted keywords as reference...")
        prompt = (
            f"<|im_start|>User:<image>Given the following image, the extracted keywords, and a question, provide a short and terse answer. "
            f"Do not paraphrase or reformat the text you see in the image. Do not include any full stops. "
            f"Just give the answer without additional explanation.\n\n"
            f"Keywords: {keywords}\n\n"
            f"Question: {question}<end_of_utterance>\nAssistant:"
        )
        final_answer = self._generate_single_response(prompt, images=image, temperature=0.0)
        print(f"  [Stage 2] Final Answer: {final_answer}")
        return final_answer
    
    def generate_inner(self, message):
        # 1. 메시지에서 이미지와 질문 파싱
        image = None
        question = ""
        for msg in message:
            if msg["type"] == "image":
                image = load_image(msg["value"])
            elif msg["type"] == "text":
                question = msg["value"].strip()
        
        if image is None or not question:
            raise ValueError("Message must contain both an image and a question.")

        # 2. Stage 1: VLM으로 키워드 추출
        extracted_keywords = self._extract_keywords(question)

        # 3. Stage 2: 이미지와 추출된 키워드를 기반으로 질문에 답변
        final_answer = self._answer_with_keywords_context(image, extracted_keywords, question)

        return {
            "extracted_keywords_stage1": extracted_keywords,
            "final_answer": final_answer,
        }
