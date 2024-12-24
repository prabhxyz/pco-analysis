"""
vision_language.py

Contains:
  1. VisionLanguageModel - a CLIP-like model combining ViT (from timm) + BERT.
  2. TechniqueAssessmentHead - predicts surgical quality indices, errors, etc.
  3. PCORiskPredictionHead - predicts PCO probability.
  4. train_vlm function for contrastive training.
  5. technique_and_pco_inference function for inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import BertTokenizer, BertModel
import timm
from tqdm import tqdm


class VisionLanguageModel(nn.Module):
    """
    A simple CLIP-like architecture that aligns:
      - Vision encoder (ViT from timm)
      - Language encoder (BERT)
    Then we do a contrastive learning objective or can produce a fused representation.
    """

    def __init__(self, vision_model_name='vit_base_patch16_224', text_model_name='bert-base-uncased', embed_dim=256):
        super().__init__()

        # Vision encoder: create a ViT from timm
        self.vision_encoder = timm.create_model(vision_model_name, pretrained=True)
        feature_dim = self.vision_encoder.num_features
        # Remove any classification head
        self.vision_encoder.reset_classifier(0)

        # Map to final embedding
        self.vision_fc = nn.Linear(feature_dim, embed_dim)

        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        text_feature_dim = self.text_encoder.config.hidden_size
        self.text_fc = nn.Linear(text_feature_dim, embed_dim)

        # Learnable logit scale for contrastive alignment
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, images):
        """
        images: (B, 3, H, W) float tensor
        returns: (B, embed_dim) float tensor
        """
        x = self.vision_encoder(images)        # (B, feature_dim)
        x = self.vision_fc(x)                 # (B, embed_dim)
        return x

    def encode_text(self, input_ids, attention_mask):
        """
        input_ids, attention_mask: from BERT tokenizer
        returns: (B, embed_dim)
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the pooler_output for classification tasks
        pooled = outputs.pooler_output  # (B, text_feature_dim)
        x = self.text_fc(pooled)        # (B, embed_dim)
        return x

    def forward(self, images, input_ids, attention_mask):
        """
        Returns contrastive logits and the image/text embeddings:
          logits_per_image, logits_per_text, image_emb, text_emb
        """
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(input_ids, attention_mask)

        # Normalize
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # Similarities
        logits_per_image = logit_scale * (image_emb @ text_emb.t())
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text, image_emb, text_emb


class TechniqueAssessmentHead(nn.Module):
    """
    Outputs a vector representing:
      - surgical quality indices
      - predicted errors
      - recommended adjustments
    (Example: 5 values total, purely for demonstration.)
    """
    def __init__(self, embed_dim=256, num_outputs=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

    def forward(self, vlm_emb):
        return self.fc(vlm_emb)


class PCORiskPredictionHead(nn.Module):
    """
    Predicts the probability of Posterior Capsule Opacification (PCO).
    Input is the fused embedding from the VisionLanguageModel.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, vlm_emb):
        return self.fc(vlm_emb)


def train_vlm(vlm_model, dataloader, device='cuda'):
    """
    A basic training loop for the Vision-Language Model using
    a CLIP-like contrastive loss on image-text pairs.
    """
    vlm_model.train()
    vlm_model.to(device)

    optimizer = optim.Adam(vlm_model.parameters(), lr=1e-5)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def clip_loss(logits_per_image, logits_per_text):
        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size, device=device)
        loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
        loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
        return (loss_img + loss_txt) / 2

    for epoch in range(1):  # Single epoch for demonstration
        print(f"[VLM Training] Epoch {epoch+1}")
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            text_prompts = batch['text_prompt']

            # Tokenize text
            encoded = tokenizer(list(text_prompts), padding=True, truncation=True, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            optimizer.zero_grad()
            logits_per_image, logits_per_text, _, _ = vlm_model(images, input_ids, attention_mask)
            loss = clip_loss(logits_per_image, logits_per_text)
            loss.backward()
            optimizer.step()

            # Break early for demonstration
            break


@torch.no_grad()
def technique_and_pco_inference(vlm_model, technique_head, pco_head, frames, text_prompts, device='cuda'):
    """
    1) Encode vision + language with the VLM
    2) Produce a fused embedding
    3) Predict technique metrics + PCO risk
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded = tokenizer(list(text_prompts), padding=True, truncation=True, return_tensors='pt')

    vlm_model.eval()
    technique_head.eval()
    pco_head.eval()

    frames = frames.to(device)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    logits_per_image, logits_per_text, image_emb, text_emb = vlm_model(frames, input_ids, attention_mask)

    # Fuse embeddings by simple average (other strategies possible)
    fused_emb = (image_emb + text_emb) / 2.0

    # Technique assessment
    tech_out = technique_head(fused_emb)
    # PCO risk
    pco_prob = pco_head(fused_emb)

    return tech_out, pco_prob
