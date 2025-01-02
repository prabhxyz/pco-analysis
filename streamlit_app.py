# streamlit_app.py
import os
import glob
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import streamlit as st
import torchvision.transforms as T

from model import SimpleSegmentationModel, NUM_CLASSES
from dataset import CLASS_NAME_TO_ID

# Same color mapping as before (index-based)
CLASS_COLORS = [
    (0, 0, 0),       # background
    (255, 0, 0),     # Iris
    (0, 255, 0),     # Pupil
    (0, 0, 255),     # Intraocular Lens
    (255, 255, 0),   # Slit/Incision Knife
    (255, 0, 255),   # Gauge
    (0, 255, 255),   # Spatula
    (128, 0, 0),     # Capsulorhexis Cystotome
    (0, 128, 0),     # Phacoemulsifier Tip
    (0, 0, 128),     # Irrigation-Aspiration
    (128, 128, 0),   # Lens Injector
    (128, 0, 128),   # Capsulorhexis Forceps
    (0, 128, 128)    # Katana Forceps
]

def overlay_mask_on_image(image_rgb, mask_np, class_toggles):
    """ Overlays color-coded segmentation mask on the original image. """
    out = image_rgb.copy()
    for cid in range(1, NUM_CLASSES):
        if not class_toggles.get(cid, True):
            continue
        color = CLASS_COLORS[cid]
        idx = (mask_np == cid)
        out[idx, 0] = 0.5 * out[idx, 0] + 0.5 * color[0]
        out[idx, 1] = 0.5 * out[idx, 1] + 0.5 * color[1]
        out[idx, 2] = 0.5 * out[idx, 2] + 0.5 * color[2]
    return out.astype(np.uint8)

def main(dataset_path="datasets/Cataract-1k/segmentation"):
    st.title("Cataract Surgery Segmentation - 12 Classes + Background")
    st.write("This app performs multi-class segmentation of surgical frames.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSegmentationModel().to(device)

    # Load weights
    try:
        model.load_state_dict(torch.load("cataract_seg_model.pth", map_location=device))
        model.eval()
    except:
        st.warning("No trained model found. Please run `python train.py` first.")
        return

    # Let user pick a video
    videos_dir = os.path.join(dataset_path, "videos")
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    if not video_files:
        st.error(f"No .mp4 files found in {videos_dir}")
        return

    selected_video = st.selectbox("Select a video:", video_files)

    # Toggle classes
    # Let's build a reverse-lookup for convenience
    # class_id -> class_name
    id_to_name = {0:"Background"}
    for k, v in CLASS_NAME_TO_ID.items():
        id_to_name[v] = k

    st.write("Toggle each class:")
    toggles = {}
    for cid in range(1, NUM_CLASSES):
        label_name = f"{cid}: {id_to_name[cid]}"
        toggles[cid] = st.checkbox(label_name, value=True)

    if selected_video:
        cap = cv2.VideoCapture(selected_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = st.slider("Frame index", 0, max(0, total_frames-1), 0, 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret:
            st.error("Could not read frame from video.")
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = frame_rgb.shape[:2]

        # Resize for model
        # Should match what's in dataset.py
        resized = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        # Convert to tensor
        tensor_img = T.ToTensor()(Image.fromarray(resized))
        tensor_img = T.Normalize(mean=(0.485,0.456,0.406),
                                 std=(0.229,0.224,0.225))(tensor_img)
        tensor_img = tensor_img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor_img)["out"]  # [1,13,512,512]
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Upscale back to original
        pred_mask_up = cv2.resize(pred_mask.astype(np.uint8),
                                  (original_w, original_h),
                                  interpolation=cv2.INTER_NEAREST)
        overlaid = overlay_mask_on_image(frame_rgb, pred_mask_up, toggles)

        st.image(overlaid, channels="RGB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/Cataract-1k/segmentation")
    args = parser.parse_args()

    main(args.dataset_path)
