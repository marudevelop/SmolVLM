from models.smolvlm import SmolVLM

class ModelName(SmolVLM):
    def answer_question(self, image, question):
        prompt = f"You are a helpful visual assistant. Given this image, think step by step, then answer the question.\n<image> {question} Let's think step by step. Answer:"
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].split("Answer:")[-1].strip()