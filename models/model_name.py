import os.path as osp
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image # PIL 이미지를 직접 다룰 수 있음
from PIL import Image

def splitlen(s, sym='/'):
    return len(s.split(sym))

class SmolVLM:
    def __init__(self, model_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct", device="cuda", **kwargs):
        # assert osp.exists(model_path) or splitlen(model_path) == 2 # 로컬 경로 또는 HuggingFace 경로 확인

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

    def build_prompt_docvqa(self, message):
        images_pil_list = []  
        question_text = ""
        blind_answer = None
        has_image_data = False 

        for msg in message:
            if msg["type"] == "image":
                if isinstance(msg["value"], Image.Image):
                    images_pil_list.append(msg["value"])
                else:
                    images_pil_list.append(load_image(msg["value"]))
                has_image_data = True
            elif msg["type"] == "text": 
                question_text = msg["value"].strip()
            elif msg["type"] == "text_blind_answer": 
                blind_answer = msg["value"]
        
        prompt_parts = ["<|im_start|>User:"] 

        if has_image_data:
            prompt_parts.append("<image>") 

        if has_image_data and blind_answer is not None:
            prompt_parts.append(
                f"The following answer was generated without access to the image and is "
                f"likely incorrect: \"{blind_answer}\". Based on the provided image, please give a short and terse answer to the following question."
                " Do not paraphrase or reformat the text you see in the image."
                " Do not include any full stops. Just give the answer without additional explanation. "
                f"Question: {question_text}"
            )
        elif has_image_data:
            # 원본 프롬프트 (이미지 + 질문만 있을 경우)
            prompt_parts.append(
                "Give a short and terse answer to the following question. "
                "Do not paraphrase or reformat the text you see in the image. Do not include any full stops. "
                "Just give the answer without additional explanation. Question: "
                f"{question_text}"
            )
        else:
            # 첫 번째 단계: 질문만 있는 경우의 프롬프트
            prompt_parts.append(
                "Give a short and terse answer to the following question. "
                "Do not include any full stops. Just give the answer without additional explanation. Question: "
                f"{question_text}"
            )
        
        prompt_parts.append("<end_of_utterance>\nAssistant:") 
        final_prompt = "".join(prompt_parts)
        
        return final_prompt, images_pil_list if images_pil_list else None

    def generate_inner(self, message, dataset=None): 
        formatted_prompt, formatted_images = self.build_prompt_docvqa(message)
        
        inputs = self.processor(
            text=formatted_prompt, images=formatted_images, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, **self.kwargs)

        input_ids_length = inputs["input_ids"].size(1) if "input_ids" in inputs else 0
        
        generated_text = self.processor.batch_decode(
            generated_ids[:, input_ids_length:], skip_special_tokens=True
        )[0]

        return generated_text.strip()