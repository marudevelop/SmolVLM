import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from collections import Counter

class SmolVLMSelfConsistency:
    def __init__(self, model_name, device, num_samples=3):
        self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device
        self.num_samples = num_samples

    def answer_question(self, image, question):
        answers = []
        cot_question = f"<image> {question} Let's think step by step."
        inputs = self.processor(text=cot_question, images=image, return_tensors="pt").to(self.device)
        for _ in range(self.num_samples):
            output = self.model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7)
            answer = self.processor.decode(output[0], skip_special_tokens=True)
            answers.append(answer)
        most_common = Counter(answers).most_common(1)[0][0]
        return most_common
