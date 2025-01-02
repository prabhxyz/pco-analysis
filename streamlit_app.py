# streamlit_app.py
import os
import glob
import cv2
import time
import tempfile
import argparse
import numpy as np
from PIL import Image

import torch
import streamlit as st
import torchvision.transforms as T

from model import FastSegmentationModel, NUM_CLASSES
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

def overlay_mask(image_rgb, mask_np, toggles):
    """Overlays color-coded segmentation mask on the original frame."""
    out = image_rgb.copy()
    for cid in range(1, NUM_CLASSES):
        if not toggles.get(cid, True):
            continue
        color = CLASS_COLORS[cid]
        idx = (mask_np == cid)
        out[idx, 0] = 0.5 * out[idx, 0] + 0.5 * color[0]
        out[idx, 1] = 0.5 * out[idx, 1] + 0.5 * color[1]
        out[idx, 2] = 0.5 * out[idx, 2] + 0.5 * color[2]
    return out.astype(np.uint8)

def main(dataset_path="datasets/Cataract-1k/segmentation"):
    st.title("Real-time Cataract Segmentation")
    st.markdown("**Play** a video and see real-time multi-class segmentation. Toggle classes below.")

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSegmentationModel(num_classes=NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load("cataract_seg_model.pth", map_location=device))
        model.eval()
    except:
        st.error("No trained model found! Please train and have 'cataract_seg_model.pth' present.")
        return

    # user picks local or upload
    choice = st.radio("Select Video Source:", ["Local Dataset", "Upload"], horizontal=True)
    video_file = None
    if choice == "Local Dataset":
        videos_dir = os.path.join(dataset_path, "videos")
        mp4s = glob.glob(os.path.join(videos_dir, "*.mp4"))
        if not mp4s:
            st.error("No mp4 in videos folder!")
            return
        video_file = st.selectbox("Choose a local video", mp4s)
    else:
        uploaded = st.file_uploader("Upload a .mp4 video", type=["mp4"])
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(uploaded.read())
            tmp.flush()
            video_file = tmp.name
        else:
            st.info("Awaiting upload...")
            return

    # define toggles for classes
    st.subheader("Toggle Classes")
    # We'll skip background=0
    id_to_name = {0: "Background"}
    for k, v in CLASS_NAME_TO_ID.items():
        id_to_name[v] = k

    toggles = {}
    cols = st.columns(6)  # 6 checkboxes per row
    # We have 12 togglable classes: let's do 2 rows if 6 columns
    i = 0
    for cid in range(1, NUM_CLASSES):
        cname = id_to_name[cid]
        with cols[i % 6]:
            toggles[cid] = st.checkbox(f"{cid}: {cname}", value=True)
        i += 1

    if not video_file:
        return

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback to 25 if missing
    st.write(f"Video has {total_frames} frames at {fps:.2f} FPS.")

    # set up session_state
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False

    # UI for controlling playback
    play_pause_btn = st.button("Play/Pause", help="Toggle real-time playback")
    if play_pause_btn:
        st.session_state.playing = not st.session_state.playing

    # manual slider if not playing
    slider_val = st.slider("Frame Index (when paused)",
                           min_value=0, max_value=total_frames-1,
                           value=st.session_state.frame_idx,
                           step=1)

    # If user drags the slider while paused, update our state
    if not st.session_state.playing:
        st.session_state.frame_idx = slider_val

    # container for images
    frame_container = st.empty()

    # If "playing" is True, we do a mini loop to read frames and update
    # We'll only do a handful of frames each run (say 1 or so), then
    # let Streamlit re-run. This is typical for "real-time" in Streamlit.
    if st.session_state.playing:
        # increment frame_idx by 1
        st.session_state.frame_idx += 1
        if st.session_state.frame_idx >= total_frames:
            st.session_state.frame_idx = 0  # loop from start

    # read the selected frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        st.error("Failed to read frame from video.")
        return

    # do inference
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = frame_rgb.shape[:2]

    resized = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    tens = T.ToTensor()(Image.fromarray(resized))
    tens = T.Normalize(mean=(0.485,0.456,0.406),
                       std=(0.229,0.224,0.225))(tens)
    tens = tens.unsqueeze(0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
        out = model(tens)["out"]
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

    # upscale mask
    pred_up = cv2.resize(pred.astype(np.uint8),
                         (original_w, original_h),
                         interpolation=cv2.INTER_NEAREST)
    # overlay
    overlaid = overlay_mask(frame_rgb, pred_up, toggles)

    frame_container.image(overlaid, channels="RGB", use_container_width=True,
                          caption=f"Frame {st.session_state.frame_idx}/{total_frames-1}")

    # If playing, sleep a bit, then rerun
    if st.session_state.playing:
        # small sleep based on FPS
        time.sleep(1.0 / fps)
        st.experimental_rerun()

def run_app():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="datasets/Cataract-1k/segmentation")
    args = parser.parse_args()
    main(args.dataset_path)

if __name__ == "__main__":
    run_app()
