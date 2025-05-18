from transformers import AutoProcessor, AutoModelForVision2Seq

class SmolVLM:
    def __init__(self, model_name, device):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)
        self.device = device

    def answer_question(self, image, question):
        prompt = f"You are a helpful visual assistant. Given this image, answer the question.\n<image> {question} Answer:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].split("Answer:")[-1].strip()