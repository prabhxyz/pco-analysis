import os
import glob
import cv2
import tempfile
import argparse
import numpy as np
from PIL import Image

import torch
import streamlit as st
import torchvision.transforms as T

from model import SimpleSegmentationModel, NUM_CLASSES
from dataset import CLASS_NAME_TO_ID

# BGR color mapping for each class ID:
CLASS_COLORS = [
    (0, 0, 0),       # Background
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

def rgb_to_hex(bgr_tuple):
    """Converts (B, G, R) into a #RRGGBB hex code."""
    b, g, r = bgr_tuple
    return f"#{r:02X}{g:02X}{b:02X}"

def overlay_mask_on_image(image_rgb, mask_np, class_toggles):
    """
    Blends segmentation masks onto the original image with 50% opacity,
    skipping any classes the user toggled off.
    """
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
    st.title("Cataract Segmentation Demo")
    st.markdown("Automatically segments **12 classes** (plus background) in cataract surgery frames.")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSegmentationModel().to(device)
    try:
        model.load_state_dict(torch.load("cataract_seg_model.pth", map_location=device))
        model.eval()
    except:
        st.error("No trained model found. Please ensure 'cataract_seg_model.pth' is in the current directory.")
        return

    # Radio button: local or upload
    choice = st.radio("Video Source:", ["Local Dataset Video", "Upload Video"], horizontal=True)
    video_file = None

    if choice == "Local Dataset Video":
        videos_dir = os.path.join(dataset_path, "videos")
        local_videos = glob.glob(os.path.join(videos_dir, "*.mp4"))
        if not local_videos:
            st.error(f"No .mp4 files found in {videos_dir}")
            return
        video_file = st.selectbox("Select a .mp4 from dataset:", local_videos)
    else:
        # Upload
        uploaded_file = st.file_uploader("Upload a .mp4", type=["mp4"])
        if uploaded_file:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(uploaded_file.read())
            tmp.flush()
            video_file = tmp.name
        else:
            st.info("Please upload a .mp4 video.")
            return

    if video_file:
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            st.error("Could not read frames from this video.")
            cap.release()
            return

        # Frame slider
        frame_idx = st.slider("Frame Index", 0, total_frames - 1, 0, 1)

        # Grab frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret:
            st.error("Failed to read frame from video.")
            return

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Prepare for model
        resized = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        in_tensor = T.ToTensor()(Image.fromarray(resized))
        in_tensor = T.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))(in_tensor)
        in_tensor = in_tensor.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(in_tensor)["out"]
            pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

        # Resize mask back to original
        pred_mask_up = cv2.resize(pred_mask.astype(np.uint8),
                                  (w, h),
                                  interpolation=cv2.INTER_NEAREST)

        # ===============================
        # Display segmented image FIRST
        # ===============================
        # We haven't toggled anything yet, so let's see toggles from last run (or default).
        # We'll define toggles in a moment, so let's do a quick hack:
        st.subheader(f"Frame {frame_idx} - Segmented Result")
        st.write("Change toggles below to show/hide classes (auto-updates).")

        # We'll define toggles after we show the image.

        # Just store the result in Session State? Alternatively, we can do
        # a two-pass approach, but let's keep it simple: We'll define toggles
        # below, but toggles are read at the top of the script if we do so.

        # Actually, let's define toggles with default (True) for everything except background.
        # Then re-run after user changes them. This is how Streamlit works, it re-runs top to bottom.
        # We'll do that after we define toggles. So let's do toggles now, after we show an empty image?

        # Actually, let's just do the toggles next, re-run the entire code:
        # That means we either need to overlay the toggles right now if they've been defined previously,
        # or define them now. We'll define them after the image, as user asked. The code will re-run in real-time.

        # For now, let's define toggles below this line,
        # but we need to know if the user changed them or not. Let's do it as second step.

        # We'll just keep a placeholder for the final image. We'll use a container to store the final image.
        segmented_image_container = st.empty()  # placeholder for the segmented image

        # =============================
        #  BELOW THE IMAGE: legend + toggles
        # =============================
        st.subheader("Color Legend")
        # We'll show each class color as a line
        id_to_name = {0: "Background"}
        for k, v in CLASS_NAME_TO_ID.items():
            id_to_name[v] = k

        for cid in range(NUM_CLASSES):
            cname = id_to_name.get(cid, f"Unknown-{cid}")
            color_hex = rgb_to_hex(CLASS_COLORS[cid])
            # Show a small colored box + class name
            st.markdown(f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px;'>"
                        f"<div style='width:16px;height:16px;background-color:{color_hex};"
                        f"border-radius:2px;'></div>"
                        f"<span style='font-size:14px;'>{cid}: {cname}</span>"
                        f"</div>", unsafe_allow_html=True)

        st.subheader("Toggle Classes On/Off")
        # toggles: background=0 usually not toggled because it's always "off" (the rest is real classes)
        toggles = {}
        for cid in range(1, NUM_CLASSES):
            cname = id_to_name.get(cid, f"Class-{cid}")
            toggles[cid] = st.checkbox(f"{cid}: {cname}", value=True)

        # Once toggles are set, overlay the mask
        final_overlay = overlay_mask_on_image(frame_rgb, pred_mask_up, toggles)
        segmented_image_container.image(final_overlay, channels="RGB", use_container_width=True)

def run_app():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/Cataract-1k/segmentation")
    args = parser.parse_args()
    main(args.dataset_path)

if __name__ == "__main__":
    run_app()
