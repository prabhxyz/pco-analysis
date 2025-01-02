#!/usr/bin/env python3
"""
real_time_demo.py

A fast, minimal OpenCV "GUI" that displays real-time segmentation with trackbar
(for frame seeking) and keyboard toggles. All usage instructions are printed
in the TERMINAL, not on the image.

Features:
  - Press SPACE to play/pause the video.
  - Use the trackbar to jump around frames.
  - Press 1..9, a, s, d to toggle classes on/off (shown in terminal).
  - Press ESC or 'q' to quit.
  - The segmentation is overlaid at alpha=0.7, so it stands out more.
  - Each toggled-on class is labeled near its largest contour in the same color
    as the overlay, with a black outline so it remains readable.

Usage:
  python real_time_demo.py --video /path/to/video.mp4 --model cataract_seg_model.pth --gpu

If you see errors about "Qt platform plugin 'xcb'", you are likely on a headless
server with no display. You can either:
  1) Install the Qt libraries and run in an environment with a visible display
  OR
  2) Use a virtual framebuffer, e.g.:  xvfb-run -a python real_time_demo.py ...

Requirements:
  pip install torch torchvision opencv-python pillow tqdm

"""

import argparse
import os
import cv2
import time
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import torch.nn as nn
import torchvision.models.segmentation as seg_models

# --------------------------
# CONFIG
# --------------------------
NUM_CLASSES = 13  # 0=Background + 12 relevant classes

# Classes
CLASS_NAME_TO_ID = {
    "Iris": 1,
    "Pupil": 2,
    "Intraocular Lens": 3,
    "Slit/Incision Knife": 4,
    "Gauge": 5,
    "Spatula": 6,
    "Capsulorhexis Cystotome": 7,
    "Phacoemulsifier Tip": 8,
    "Irrigation-Aspiration": 9,
    "Lens Injector": 10,
    "Capsulorhexis Forceps": 11,
    "Katana Forceps": 12
}
ID_TO_NAME = {0: "Background"}
for k, v in CLASS_NAME_TO_ID.items():
    ID_TO_NAME[v] = k

# BGR color mapping
CLASS_COLORS = [
    (0, 0, 0),       # 0 background
    (255, 0, 0),     # 1
    (0, 255, 0),     # 2
    (0, 0, 255),     # 3
    (255, 255, 0),   # 4
    (255, 0, 255),   # 5
    (0, 255, 255),   # 6
    (128, 0, 0),     # 7
    (0, 128, 0),     # 8
    (0, 0, 128),     # 9
    (128, 128, 0),   # 10
    (128, 0, 128),   # 11
    (0, 128, 128)    # 12
]

# --------------------------
# MODEL
# --------------------------
class FastSegmentationModel(nn.Module):
    """
    Deeplabv3 + MobileNetV3-Large, pretrained on COCO,
    with final layer replaced to produce NUM_CLASSES channels.
    """
    def __init__(self):
        super().__init__()
        self.model = seg_models.deeplabv3_mobilenet_v3_large(pretrained=True)
        # Replace final classifier conv to match our classes
        self.model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        return self.model(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to .mp4")
    parser.add_argument("--model", default="cataract_seg_model.pth", help="Path to .pth model")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--width", type=int, default=512, help="Resize input width for the model")
    parser.add_argument("--height", type=int, default=512, help="Resize input height for the model")
    return parser.parse_args()

def nothing(x):
    """ Trackbar callback (no-op). """
    pass

def find_largest_contour(seg_mask, cid):
    """
    Return centroid (x, y) of largest contour for pixels == cid in seg_mask.
    If none, return None.
    """
    roi = (seg_mask == cid).astype(np.uint8)
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None

def overlay_mask(img_rgb, seg_mask, toggles, alpha=0.7):
    """
    Overlays seg_mask on img_rgb with alpha blending for toggled classes.
    Return the result in RGB.
    """
    out = img_rgb.copy()
    h, w = out.shape[:2]
    for cid in range(1, NUM_CLASSES):
        if not toggles.get(cid, True):
            continue
        b,g,r = CLASS_COLORS[cid]
        idx = (seg_mask == cid)
        out[idx,0] = (1 - alpha)*out[idx,0] + alpha*b
        out[idx,1] = (1 - alpha)*out[idx,1] + alpha*g
        out[idx,2] = (1 - alpha)*out[idx,2] + alpha*r
    return out

def label_mask(bgr_img, seg_mask, toggles):
    """
    For each toggled class, label the largest contour in the same color as the overlay.
    We'll do a small rectangle with text in that color, plus black outline.
    """
    for cid in range(1, NUM_CLASSES):
        if not toggles.get(cid, True):
            continue
        center = find_largest_contour(seg_mask, cid)
        if center is None:
            continue
        x, y = center
        color = CLASS_COLORS[cid]  # (b,g,r)
        class_name = ID_TO_NAME.get(cid, f"Class {cid}")
        text_size = 0.5
        thickness = 2

        # Draw a small rectangle as background in the same color
        # (some might prefer black bg or semitransparent).
        # We'll do a black outline around text for visibility:
        (txt_w, txt_h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, text_size, thickness)
        rect_w = txt_w + 6
        rect_h = txt_h + 6

        # Top-left corner for the rectangle
        rx1, ry1 = x, y - rect_h
        rx2, ry2 = x + rect_w, y
        # clip if out of bounds
        if rx1 < 0: rx1 = 0
        if ry1 < 0: ry1 = 0

        cv2.rectangle(bgr_img, (rx1, ry1), (rx2, ry2), color, -1)
        # put text
        # to put text inside the rectangle, offset a bit
        tx = rx1 + 3
        ty = ry1 + txt_h
        # Outline first (black)
        cv2.putText(bgr_img, class_name, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    (0,0,0), thickness+2)
        # Then main text in the color
        cv2.putText(bgr_img, class_name, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size,
                    color, thickness)
    return bgr_img

def print_instructions():
    """ Print usage instructions in the terminal. """
    lines = [
        "",
        "================= USAGE =================",
        "Press SPACE to toggle PLAY/PAUSE.",
        "Drag the 'Frame' trackbar to jump around in time when paused.",
        "Toggle classes with these keys: '1'..'9','a','s','d'.",
        "Press ESC or 'q' to quit at any time.",
        "========================================="
    ]
    print("\n".join(lines))

def print_toggles(toggles):
    """ Print toggles for each class in the terminal. """
    print("Class toggles:")
    for cid in range(1, NUM_CLASSES):
        status = "ON " if toggles[cid] else "OFF"
        cname = ID_TO_NAME.get(cid, f"Class {cid}")
        print(f"  {cid}: {cname} = {status}")
    print("")

def main():
    args = parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model
    model = FastSegmentationModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # class toggles
    toggles = {cid: True for cid in range(1, NUM_CLASSES)}

    # key->class_id map
    toggle_key_map = {
        ord('1'): 1, ord('2'): 2, ord('3'): 3, ord('4'): 4, ord('5'): 5,
        ord('6'): 6, ord('7'): 7, ord('8'): 8, ord('9'): 9,
        ord('a'): 10, ord('s'): 11, ord('d'): 12
    }

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Failed to open video:", args.video)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video: {args.video}, frames={total_frames}, fps={fps:.2f}")

    # Show instructions in terminal once
    print_instructions()
    # Show toggles in terminal initially
    print_toggles(toggles)

    playing = False
    current_frame_idx = 0

    cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Segmented", 1280, 720)
    cv2.createTrackbar("Frame", "Segmented", 0, max(0,total_frames-1), nothing)

    # For half-precision speed
    autocast_enabled = (device.type == "cuda")

    while True:
        # If paused, follow trackbar. If playing, set trackbar to current frame
        track_pos = cv2.getTrackbarPos("Frame", "Segmented")
        if not playing:
            current_frame_idx = track_pos
        else:
            cv2.setTrackbarPos("Frame", "Segmented", current_frame_idx)

        # wrap around if past last frame
        if current_frame_idx >= total_frames:
            current_frame_idx = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            # End of video or error
            current_frame_idx = 0
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = frame_rgb.shape[:2]

        # Preprocess
        resized = cv2.resize(frame_rgb, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        tens = T.ToTensor()(Image.fromarray(resized))
        tens = T.Normalize(mean=(0.485,0.456,0.406),
                           std=(0.229,0.224,0.225))(tens)
        tens = tens.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enabled):
            out = model(tens)["out"]
            pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

        # Resize mask to original size
        pred_up = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h), cv2.INTER_NEAREST)

        # Overlay
        overlay_rgb = overlay_mask(frame_rgb, pred_up, toggles, alpha=0.7)
        overlaid_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

        # Label largest contour for toggled classes in the same color
        overlaid_bgr = label_mask(overlaid_bgr, pred_up, toggles)

        # Show
        cv2.imshow("Segmented", overlaid_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("Exiting...")
            break
        elif key == 32:  # Space
            playing = not playing
            print(f"Playing={playing}")
        elif key in toggle_key_map:
            cid = toggle_key_map[key]
            toggles[cid] = not toggles[cid]
            print_toggles(toggles)

        if playing:
            current_frame_idx += 1
            time.sleep(1.0 / fps)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
