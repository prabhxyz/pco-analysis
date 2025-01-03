import torch
from transformers import AutoTokenizer

class FeedbackGenerator:
    """
    Uses a VLM model to produce textual feedback for each recognized phase.
    Compares the current frame's embedding to a set of text prompts about best practices or warnings.
    """
    def __init__(self, vlm_model, device="cpu"):
        self.vlm = vlm_model.to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Each phase => list of textual statements
        self.prompts_map = {
            "Irrigation/Aspiration": [
                "Complete removal of cortical material is performed",
                "There might be residual lens epithelial cells"
            ],
            "Capsule Polishing": [
                "The surgeon thoroughly polishes the capsule to reduce PCO",
                "The polishing step is incomplete or rushed"
            ]
            # Add more if desired
        }

    def generate_phase_feedback(self, frame_embed, phase_name):
        if phase_name in self.prompts_map:
            prompts = self.prompts_map[phase_name]
        else:
            prompts = [
                f"{phase_name} is performed with correct technique",
                f"{phase_name} might have suboptimal approach"
            ]

        # Encode each prompt, measure similarity
        all_sims = []
        for prompt_text in prompts:
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            text_emb = self.vlm.encode_text(**inputs)  # [1,256]
            sim_val = torch.sum(frame_embed * text_emb, dim=-1).item()
            all_sims.append(sim_val)

        # Pick best prompt
        best_idx = int(torch.argmax(torch.tensor(all_sims)))
        best_prompt = prompts[best_idx]
        return f"[VLM] Phase '{phase_name}': {best_prompt}"

    def get_frame_feedback(self, pixel_values, phase_name):
        """
        pixel_values: shape [1,3,224,224]
        returns a feedback string
        """
        with torch.no_grad():
            image_emb = self.vlm.encode_image(pixel_values)  # [1,256]
        feedback_str = self.generate_phase_feedback(image_emb[0], phase_name)
        return feedback_str