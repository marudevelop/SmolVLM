import os.path as osp
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from PIL import Image

def splitlen(s, sym='/'):
    return len(s.split(sym))

class SmolVLM_ft:
    def __init__(self, model_path="/data3/jykim/Projects/VQA/SmolVLM/checkpoints/stage1/checkpoint-139532", device="cuda", **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.device = device

        # 파인튜닝된 모델의 경우 베이스 모델에서 processor를 로드
        # 또는 파인튜닝 과정에서 processor도 저장되었다면 model_path에서 로드
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
        except:
            # processor가 체크포인트에 없는 경우 베이스 모델에서 로드
            self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
        
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
        prompt = (
            "<|im_start|>user\n"
            "Please think step by step and provide the bounding box coordinate of the region that can help you answer the question better.\n"
            "<image>\n"
            f"Question: {message}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        images = []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
        
        return prompt, images
    
    def generate_inner(self, message, dataset=None):
        formatted_messages, formatted_images = self.build_prompt_docvqa(message)

        images = (
            [formatted_images]
            if isinstance(formatted_images, Image.Image)
            else formatted_images
        )

        inputs = self.processor(
            text=formatted_messages, images=images, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, **self.kwargs)

        generated_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
        )[0]

        return generated_text.strip()