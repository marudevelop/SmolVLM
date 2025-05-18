from transformers import AutoProcessor, AutoModelForVision2Seq

class SmolVLM:
    def __init__(self, model_name, device):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)
        self.device = device
        self.last_choices_for_parsing = []

    def answer_question(self, image, question_with_choices):
        prompt = (
            f"You are a helpful visual assistant. Given the image, question, and choices, "
            f"please select the correct answer and respond with ONLY the letter of the correct choice (e.g., A, B, C, D).\n"
            f"<image>\n"
            f"{question_with_choices}\n"
            f"Answer:"
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=10)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        answer_part = generated_text.split("Answer:")[-1].strip()
        if answer_part:
            return answer_part[0].upper()
        return ""
