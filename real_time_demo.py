"""
Demonstration of real-time surgical video guidance:
 - Displays segmentation outlines for instruments.
 - Recognizes the current surgical phase each frame.
 - Prints technique feedback in the terminal.
 - Displays PCO risk and phase in the top-left corner of the GUI.
 - Has a trackbar to scrub video frames, plus play/pause support.
 - Allows an optional chat mode in the terminal if '--chat' is used.

Usage:
  python real_time_demo.py --video /path/to/video.mp4 \
                           --seg_model lightweight_seg.pth \
                           --phase_model phase_recognition.pth \
                           --chat
"""

import argparse
import cv2
import time
import torch
import torch.nn.functional as F
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from training.models import LightweightSegModel, PhaseRecognitionNet
from technique_assessment import TechniqueAdvisor
from pco_assessment import PCORiskAssessor
from assistant import SurgeryAssistant

def nothing(x):
    pass

INSTRUMENT_NAMES = [
    "Background",
    "Iris",
    "Pupil",
    "IntraocularLens",
    "SlitKnife",
    "Gauge",
    "Spatula",
    "CapsulorhexisCystotome",
    "PhacoTip",
    "IrrigationAspiration",
    "LensInjector",
    "CapsulorhexisForceps",
    "KatanaForceps"
]

INSTRUMENT_COLORS = [
    (0,0,0),
    (255,0,0),
    (0,255,0),
    (0,0,255),
    (255,255,0),
    (255,0,255),
    (0,255,255),
    (128,0,0),
    (0,128,0),
    (0,0,128),
    (128,128,0),
    (128,0,128),
    (0,128,128)
]

def find_contours_for_class(seg_mask, class_id):
    bin_mask = (seg_mask == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_segmentation_outlines(bgr_img, seg_mask):
    for cid in np.unique(seg_mask):
        if cid == 0:
            continue
        color = INSTRUMENT_COLORS[cid]
        contours = find_contours_for_class(seg_mask, cid)
        cv2.drawContours(bgr_img, contours, -1, color, 2)
    return bgr_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 or camera index.")
    parser.add_argument("--seg_model", type=str, default="lightweight_seg.pth")
    parser.add_argument("--phase_model", type=str, default="phase_recognition.pth")
    parser.add_argument("--chat", action="store_true", help="Enter chat mode in terminal if desired.")
    parser.add_argument("--no_cuda", action="store_true", help="Use CPU only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    seg_model = LightweightSegModel(num_classes=13, use_pretrained=False, aux_loss=True).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model, map_location=device), strict=False)
    seg_model.eval()

    phase_model = PhaseRecognitionNet(num_phases=12, use_pretrained=True).to(device)
    phase_model.load_state_dict(torch.load(args.phase_model, map_location=device), strict=True)
    phase_model.eval()

    technique_advisor = TechniqueAdvisor()
    pco_assessor = PCORiskAssessor()
    assistant = SurgeryAssistant()

    seg_transform = A.Compose([
        A.Resize(512,512),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    phase_transform = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    source = 0 if args.video is None else args.video
    if str(source).isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open video/camera: {source}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else 0
    print(f"Total frames: {total_frames}" if total_frames>0 else "Unknown total frames (possibly camera).")

    cv2.namedWindow("SurgeryDemo", cv2.WINDOW_NORMAL)
    if total_frames > 0:
        cv2.createTrackbar("Frame", "SurgeryDemo", 0, total_frames-1, nothing)

    recognized_phase_history = []
    playing = False
    current_frame_index = 0

    print("Controls:")
    print("  SPACE or 'p': Toggle Play/Pause")
    print("  'c': Enter chat mode (if --chat)")
    print("  'q': Quit")
    print("  Trackbar: Scrub frames in the video (when paused).")

    while True:
        if playing:
            current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            if total_frames > 0:
                track_pos = cv2.getTrackbarPos("Frame", "SurgeryDemo")
                current_frame_index = track_pos
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)

        ret, frame_bgr = cap.read()
        if not ret:
            if total_frames > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame_index = 0
            else:
                print("End of stream or error.")
                break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Segmentation
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            seg_in = seg_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            seg_out_dict = seg_model(seg_in)
            seg_logits = seg_out_dict["out"]
            seg_pred = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

        overlay_bgr = frame_bgr.copy()
        overlay_bgr = draw_segmentation_outlines(overlay_bgr, seg_pred)

        # Phase
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            phase_in = phase_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            phase_logits = phase_model(phase_in)
            phase_label = int(torch.argmax(phase_logits, dim=1).item())

        # Basic phase map (0..11)
        phase_map = {
            0:"Incision", 1:"Viscoelastic", 2:"Capsulorhexis", 3:"Hydrodissection",
            4:"Phacoemulsification", 5:"Irrigation/Aspiration", 6:"Capsule Polishing",
            7:"Lens Implantation", 8:"Lens positioning", 9:"Viscoelastic Suction",
            10:"Tonifying/Antibiotics", 11:"Other"
        }
        phase_name = phase_map.get(phase_label, "Unknown")
        recognized_phase_history.append(phase_name)

        # Technique
        technique_text = technique_advisor.get_feedback(seg_pred, phase_name)
        if technique_text.strip():
            print("[TechniqueAdvisor]:", technique_text)

        # PCO
        pco_risk = pco_assessor.estimate_risk(seg_pred, recognized_phase_history)

        # Chat context
        present_ids = np.unique(seg_pred)
        instr_names = []
        for cid in present_ids:
            if cid != 0:
                instr_names.append(INSTRUMENT_NAMES[cid])
        assistant.update_context(phase_name, pco_risk, instr_names)

        # Display phase + pco
        cv2.putText(overlay_bgr, f"Phase: {phase_name}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)
        cv2.putText(overlay_bgr, f"Phase: {phase_name}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)

        cv2.putText(overlay_bgr, f"PCO Risk: {pco_risk}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)
        cv2.putText(overlay_bgr, f"PCO Risk: {pco_risk}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),1)

        if playing and total_frames > 0:
            cv2.setTrackbarPos("Frame", "SurgeryDemo", current_frame_index)

        cv2.imshow("SurgeryDemo", overlay_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('p') or key == 32:  # 'p' or SPACE
            playing = not playing
            print(f"Playing={playing}")
        elif key == ord('c') and args.chat:
            user_txt = input("Chat (type 'done' to exit): ")
            while user_txt.lower() != "done":
                ans = assistant.answer(user_txt)
                print("[Assistant]:", ans)
                user_txt = input("> ")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()