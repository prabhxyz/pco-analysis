#!/usr/bin/env python3
"""
real_time_demo.py

An OpenCV-based segmentation viewer for your cataract model with:
 - A trackbar to jump to any frame
 - Real-time toggles for each class (1..9, a, s, d)
 - Press SPACE to toggle play/pause
 - Press ESC/q to quit

Usage:
  python real_time_demo.py --video /path/to/video.mp4 --model cataract_seg_model.pth [--gpu]

Requirements:
  pip install torch torchvision opencv-python pillow tqdm
  (plus GPU drivers if using --gpu)

Fixing Qt plugin errors:
  sudo apt-get install -y libqt5gui5 libqt5core5a libqt5widgets5 libqt5network5 libqt5test5 libxkbcommon-x11-0 libxcb-xinerama0
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

NUM_CLASSES = 13  # 0=Background + 12 classes

class FastSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = seg_models.deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    def forward(self, x):
        return self.model(x)

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
    (0,0,0),      # 0
    (255,0,0),    # 1
    (0,255,0),    # 2
    (0,0,255),    # 3
    (255,255,0),  # 4
    (255,0,255),  # 5
    (0,255,255),  # 6
    (128,0,0),    # 7
    (0,128,0),    # 8
    (0,0,128),    # 9
    (128,128,0),  # 10
    (128,0,128),  # 11
    (0,128,128)   # 12
]

def overlay_mask_on_image(img_rgb, mask, toggles):
    out = img_rgb.copy()
    for cid in range(1, NUM_CLASSES):
        if not toggles.get(cid, True):
            continue
        b,g,r = CLASS_COLORS[cid]
        idx = (mask == cid)
        out[idx, 0] = 0.5*out[idx, 0] + 0.5*b
        out[idx, 1] = 0.5*out[idx, 1] + 0.5*g
        out[idx, 2] = 0.5*out[idx, 2] + 0.5*r
    return out.astype(np.uint8)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to .mp4 video")
    parser.add_argument("--model", default="cataract_seg_model.pth", help="Path to .pth model file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--width", type=int, default=512, help="Model input width")
    parser.add_argument("--height", type=int, default=512, help="Model input height")
    return parser.parse_args()

def nothing(x):
    # Callback for createTrackbar, does nothing
    pass

def main():
    args = parse_args()
    device = torch.device("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    # load model
    model = FastSegmentationModel().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # toggles for each class ID
    toggles = {cid: True for cid in range(1, NUM_CLASSES)}

    # map keys to class IDs
    toggle_key_map = {
        ord('1'): 1, ord('2'): 2, ord('3'): 3, ord('4'): 4, ord('5'): 5,
        ord('6'): 6, ord('7'): 7, ord('8'): 8, ord('9'): 9,
        ord('a'): 10, ord('s'): 11, ord('d'): 12
    }

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error opening video:", args.video)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video info: {total_frames} frames, ~{fps:.2f} fps")

    # We'll keep track of current frame in a variable
    # We'll also keep a "playing" bool
    current_frame_idx = 0
    playing = False

    # Create a window and a trackbar for "time"
    cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Segmented", 1280, 720)
    cv2.createTrackbar("Frame", "Segmented", 0, max(0, total_frames-1), nothing)

    # Half-precision on GPU for speed
    autocast_enabled = (device.type == "cuda")

    while True:
        # If user is dragging the trackbar, update current_frame_idx
        track_pos = cv2.getTrackbarPos("Frame", "Segmented")
        if not playing:
            # If not playing, we follow the trackbar
            current_frame_idx = track_pos
        else:
            # If playing, we override the trackbar with our current_frame_idx
            cv2.setTrackbarPos("Frame", "Segmented", current_frame_idx)

        if current_frame_idx >= total_frames:
            current_frame_idx = 0  # loop from start

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            print("End of video or error.")
            current_frame_idx = 0
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]

        # Prepare for model
        resized = cv2.resize(frame_rgb, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
        tens = T.ToTensor()(Image.fromarray(resized))
        tens = T.Normalize(mean=(0.485,0.456,0.406),
                           std=(0.229,0.224,0.225))(tens)
        tens = tens.unsqueeze(0).to(device)

        # Inference
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=autocast_enabled):
            output = model(tens)["out"]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Upscale
        pred_up = cv2.resize(pred.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)

        # Overlay
        overlaid = overlay_mask_on_image(frame_rgb, pred_up, toggles)
        overlaid_bgr = cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR)

        # UI overlay text
        instructions = "Keys 1..9,a,s,d = Toggle classes 1..12 | SPACE=play/pause | ESC/q=quit"
        cv2.putText(overlaid_bgr, instructions, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        # Show toggles in top-left corner
        y_offset = 70
        for cid in range(1, NUM_CLASSES):
            status = "ON " if toggles[cid] else "OFF"
            text = f"{cid}:{ID_TO_NAME[cid]} {status}"
            cv2.putText(overlaid_bgr, text, (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            y_offset += 20

        # Show the result
        cv2.imshow("Segmented", overlaid_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("Exiting...")
            break
        elif key == 32:  # SPACE
            playing = not playing
            print("Playing:", playing)
        elif key in toggle_key_map:
            cid = toggle_key_map[key]
            toggles[cid] = not toggles[cid]
            print(f"Toggled class {cid} -> {toggles[cid]}")

        if playing:
            # Move forward in time
            current_frame_idx += 1
            time.sleep(1.0/fps)  # wait to match real-time playback
        # else we remain paused at the trackbar's position

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
