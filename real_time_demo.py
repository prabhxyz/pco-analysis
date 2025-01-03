"""
Demonstration:
 - Real-time segmentation outlines.
 - Phase recognition.
 - A fast, naive PCO snapshot each frame (no lag).
 - A chat interface that can trigger an LLM-based PCO analysis on demand.

Files that can be deleted from older code:
 - pco_assessment.py
 - pco_prompt_assessment.py

All PCO logic is now inside pco_intelligence.py.
"""

import argparse
import cv2
import time
import torch
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from training.models import LightweightSegModel, PhaseRecognitionNet
from technique_assessment import TechniqueAdvisor
from pco_intelligence import PCOIntelligence
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
    mask_bin = (seg_mask == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_segmentation_outlines(bgr_img, seg_mask):
    for cid in np.unique(seg_mask):
        if cid == 0:
            continue
        color = INSTRUMENT_COLORS[cid]
        ctrs = find_contours_for_class(seg_mask, cid)
        cv2.drawContours(bgr_img, ctrs, -1, color, 2)
    return bgr_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--seg_model", type=str, default="lightweight_seg.pth")
    parser.add_argument("--phase_model", type=str, default="phase_recognition.pth")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--llm_model", type=str, default=None,
                        help="If set, loads a text-generation pipeline for PCO analysis in chat. e.g. gpt2")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Load segmentation
    seg_model = LightweightSegModel(num_classes=13, use_pretrained=False, aux_loss=True).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model, map_location=device), strict=False)
    seg_model.eval()

    # Load phase model
    phase_model = PhaseRecognitionNet(num_phases=12, use_pretrained=True).to(device)
    phase_model.load_state_dict(torch.load(args.phase_model, map_location=device), strict=True)
    phase_model.eval()

    # Technique advisor
    technique_advisor = TechniqueAdvisor()

    # Initialize combined naive + LLM approach
    llm_name = args.llm_model if args.llm_model else None
    intelligence = PCOIntelligence(llm_model_name=llm_name, device=device.type)

    # Assistant
    assistant = SurgeryAssistant(pco_intelligence=intelligence)

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

    # Video/camera
    source = 0 if args.video is None else args.video
    if str(source).isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open: {source}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT)>0 else 0
    if total_frames>0:
        print(f"Total frames: {total_frames}")
    else:
        print("Unknown total frames (camera or issue).")

    cv2.namedWindow("SurgeryDemo", cv2.WINDOW_NORMAL)
    if total_frames>0:
        cv2.createTrackbar("Frame", "SurgeryDemo", 0, total_frames-1, nothing)

    recognized_phase_history = []
    playing = False
    current_frame_idx = 0

    print("Controls:")
    print("  SPACE or 'p': Toggle Play/Pause")
    print("  'q': Quit")
    print("  'c': Terminal chat (if --chat)")
    print("  Trackbar: Scrub frames in video (when paused).")

    while True:
        if playing:
            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            if total_frames>0:
                tb_pos = cv2.getTrackbarPos("Frame", "SurgeryDemo")
                current_frame_idx = tb_pos
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

        ret, frame_bgr = cap.read()
        if not ret:
            if total_frames>0:
                # wrap around
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame_idx = 0
            else:
                print("No more frames or read error.")
                break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Seg inference
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            seg_in = seg_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            seg_out = seg_model(seg_in)["out"]
            seg_pred = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()

        overlay_bgr = frame_bgr.copy()
        overlay_bgr = draw_segmentation_outlines(overlay_bgr, seg_pred)

        # Phase inference
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            ph_in = phase_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            ph_logits = phase_model(ph_in)
            ph_label = int(torch.argmax(ph_logits, dim=1).item())

        phase_map = {
            0:"Incision", 1:"Viscoelastic", 2:"Capsulorhexis", 3:"Hydrodissection",
            4:"Phacoemulsification", 5:"Irrigation/Aspiration", 6:"Capsule Polishing",
            7:"Lens Implantation", 8:"Lens positioning", 9:"Viscoelastic Suction",
            10:"Tonifying/Antibiotics", 11:"Other"
        }
        phase_name = phase_map.get(ph_label, "Unknown")

        if phase_name not in recognized_phase_history:
            recognized_phase_history.append(phase_name)

        # Technique feedback in terminal
        tech_text = technique_advisor.get_feedback(seg_pred, phase_name)
        if tech_text.strip():
            print("[TechniqueAdvisor]:", tech_text)

        # Fast snapshot
        pco_risk = intelligence.realtime_snapshot(seg_pred, recognized_phase_history)

        # Update assistant
        present_ids = np.unique(seg_pred)
        instr = [INSTRUMENT_NAMES[c] for c in present_ids if c!=0]
        assistant.update_context(phase_name, pco_risk, instr)

        # Show minimal text
        cv2.putText(overlay_bgr, f"Phase: {phase_name}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)
        cv2.putText(overlay_bgr, f"Phase: {phase_name}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)

        cv2.putText(overlay_bgr, f"PCO: {pco_risk}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)
        cv2.putText(overlay_bgr, f"PCO: {pco_risk}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),1)

        if playing and total_frames>0:
            cv2.setTrackbarPos("Frame", "SurgeryDemo", current_frame_idx)

        cv2.imshow("SurgeryDemo", overlay_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting.")
            break
        elif key == ord('p') or key == 32:  # SPACE
            playing = not playing
            print(f"Playing={playing}")
        elif key == ord('c') and args.chat:
            # Chat loop
            user_txt = input("Ask about surgery (type 'done' to exit): ")
            while user_txt.lower() != "done":
                ans = assistant.answer(user_txt)
                print("[Assistant]:", ans)
                user_txt = input("> ")

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()