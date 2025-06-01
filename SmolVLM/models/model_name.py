### 이거 그냥 SmolVLM.py 코드 복사 붙여넣기 한거

import os.path as osp
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from PIL import Image

def splitlen(s, sym='/'):
    return len(s.split(sym))

class ModelName:
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

    def build_prompt_docvqa(self, message):
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