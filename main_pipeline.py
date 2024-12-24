"""
main_pipeline.py

This script does the following:
  1. Loading the cataract-1k dataset
  2. Training the segmentation model (Part a)
  3. Training the vision-language model (Part b)
  4. Running an example inference flow
     - Segmentation -> Optical flow -> Region-based tracking
     - Vision-Language -> Technique assessment + PCO risk
"""

import torch
from cataract_dataset import get_cataract_dataloaders
from segmentation_tracking import (
    SimpleSegmentationModel,
    train_segmentation_model,
    run_segmentation_inference,
    compute_optical_flow,
    region_based_tracking
)
from vision_language import (
    VisionLanguageModel,
    TechniqueAssessmentHead,
    PCORiskPredictionHead,
    train_vlm,
    technique_and_pco_inference
)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 1) Load Data ---
    train_loader, val_loader = get_cataract_dataloaders(root_dir='./cataract-1k', batch_size=4)

    # --- 2) Initialize Models ---

    # Part (a) Segmentation
    seg_model = SimpleSegmentationModel(num_classes=2).to(device)

    # Part (b) VLM
    vlm_model = VisionLanguageModel(
        vision_model_name='vit_base_patch16_224',
        text_model_name='bert-base-uncased',
        embed_dim=256
    ).to(device)

    # Technique & PCO heads
    technique_head = TechniqueAssessmentHead(embed_dim=256, num_outputs=5).to(device)
    pco_head = PCORiskPredictionHead(embed_dim=256).to(device)

    # --- 3) TRAINING (Demo) ---

    print("\n=== Training Segmentation Model (Part a) ===")
    train_segmentation_model(seg_model, train_loader, device=device)

    print("\n=== Training Vision-Language Model (Part b) ===")
    train_vlm(vlm_model, train_loader, device=device)

    # --- 4) INFERENCE EXAMPLE ---

    print("\n=== Inference Example ===")
    val_batch = next(iter(val_loader))
    images = val_batch['image']       # (B, 3, 224, 224)
    text_prompts = val_batch['text_prompt']

    # (a) Segmentation
    seg_preds = run_segmentation_inference(seg_model, images, device=device)

    # (a) Optical Flow (example: just compute flow between two consecutive frames in batch)
    # We only compute flow between images[0] and images[1] for demonstration if batch size >=2
    if images.shape[0] >= 2:
        # Convert first two images to CPU NumPy for OpenCV
        img0_np = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        img1_np = (images[1].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        flow = compute_optical_flow(img0_np, img1_np)
        print(f"[Optical Flow] Flow shape: {flow.shape}")
    else:
        print("[Optical Flow] Not enough frames to compute flow in this batch.")

    # (a) Region-based tracking
    tracked_info = region_based_tracking(seg_preds)
    print("[Tracking] Centroids of largest region in each mask:", tracked_info)

    # (b) Vision-Language + technique + PCO
    tech_out, pco_prob = technique_and_pco_inference(
        vlm_model, technique_head, pco_head, images, text_prompts, device=device
    )
    print("[Technique Assessment Output]:", tech_out)
    print("[PCO Risk Probability]:", pco_prob)

    print("\nAll done!")


if __name__ == "__main__":
    main()
