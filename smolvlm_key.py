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
            f"<|im_start|>User:Extract the most relevant comma-separated keywords from the following question. Do not include any other text or punctuation.\n\n"
            f"Question: {question}<end_of_utterance>\nAssistant:"
        )

        extracted_keywords = self._generate_single_response(
            prompt, 
            images=None, 
            temperature=0.0,
            max_new_tokens=50, # 키워드는 길지 않으므로 최대 토큰 제한
            no_repeat_ngram_size=2, # 2-gram 반복 방지
            early_stopping=True # EOS 토큰 생성 시 즉시 중단
        )
        print(f"  [Stage 1] Extracted keywords: {extracted_keywords}")
        return extracted_keywords

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