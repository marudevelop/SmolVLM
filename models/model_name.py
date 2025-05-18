from models.smolvlm import SmolVLM

class ModelName(SmolVLM):
    def answer_question(self, image, question_with_choices):
        prompt = (
            f"You are a helpful visual assistant. Given the image, question, and choices, "
            f"think step by step to determine the correct answer. "
            f"After your reasoning, conclude with 'The final answer is [LETTER]' where [LETTER] is the letter of the correct choice (e.g., A, B, C, D).\n"
            f"<image>\n"
            f"{question_with_choices}\n"
            f"Let's think step by step. The final answer is "
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

        answer_part = ""
        if "The final answer is" in generated_text:
            answer_segment = generated_text.split("The final answer is")[-1].strip()
            if answer_segment:
                potential_answer = answer_segment[0]
                if 'A' <= potential_answer.upper() <= 'Z':
                    answer_part = potential_answer.upper()
        
        if not answer_part and "Answer:" in generated_text:
            answer_segment = generated_text.split("Answer:")[-1].strip()
            if answer_segment:
                potential_answer = answer_segment[0]
                if 'A' <= potential_answer.upper() <= 'Z':
                    answer_part = potential_answer.upper()

        if not answer_part:
            for char_idx in range(len(generated_text) -1, -1, -1):
                char = generated_text[char_idx]
                if 'A' <= char.upper() <= 'Z':
                    answer_part = char.upper()
                    break
        
        return answer_part
