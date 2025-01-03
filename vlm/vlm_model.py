import torch
import torch.nn as nn
from transformers import ViTModel, DistilBertModel

class VLMModel(nn.Module):
    """
    A simplified CLIP-like model with:
      - Vision encoder: ViT
      - Text encoder: DistilBert
    Aligned in a shared embedding dimension (256).
    """
    def __init__(self, vision_model_name="google/vit-base-patch16-224", text_model_name="distilbert-base-uncased"):
        super().__init__()
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)

        # projection heads to 256
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, 256)
        self.text_proj = nn.Linear(self.text_encoder.config.dim, 256)

    def encode_image(self, pixel_values):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        cls_embed = outputs.last_hidden_state[:,0]  # [B,hid]
        proj = self.vision_proj(cls_embed)          # [B,256]
        return proj

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:,0]
        proj = self.text_proj(cls_token)
        return proj