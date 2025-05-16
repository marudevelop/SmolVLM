import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

class SmolVLM:
    def __init__(self, model_name, device):
        self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device

    def answer_question(self, image, question):
        question = f"<image> {question}"
        inputs = self.processor(text=question, images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=500)
        return self.processor.decode(output[0], skip_special_tokens=True)
