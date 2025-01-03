"""
A pipeline to:
  1) Segment instruments with LightweightSegModel (DeepLab + MobileNetV3).
  2) Recognize phase with PhaseRecognitionNet (MobileNetV3).
  3) Generate feedback from:
     - TechniqueAdvisor (heuristic checks)
     - VLM-based approach (ViT + DistilBERT).
  4) Store feedback with timestamps, skip duplicates, print final summary.

Usage in one line:
  python real_time_demo.py \
    --video training/datasets/Cataract-1k/segmentation/videos/case_5016.mp4 \
    --seg_model lightweight_seg.pth \
    --phase_model phase_recognition.pth
"""

import argparse
import cv2
import torch
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from training.models import LightweightSegModel, PhaseRecognitionNet
from vlm.vlm_model import VLMModel
from vlm.feedback_generator import FeedbackGenerator
from technique_assessment import TechniqueAdvisor

def nothing(x):
    pass

PHASE_MAP = {
    0:"Incision",1:"Viscoelastic",2:"Capsulorhexis",3:"Hydrodissection",
    4:"Phacoemulsification",5:"Irrigation/Aspiration",6:"Capsule Polishing",
    7:"Lens Implantation",8:"Lens positioning",9:"Viscoelastic Suction",
    10:"Tonifying/Antibiotics",11:"Other"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--seg_model", type=str, default="lightweight_seg.pth")
    parser.add_argument("--phase_model", type=str, default="phase_recognition.pth")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Using device:", device)

    # 1) Load segmentation model (LightweightSegModel)
    seg_model = LightweightSegModel(num_classes=13, use_pretrained=False, aux_loss=True).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model, map_location=device), strict=False)
    seg_model.eval()

    # 2) Phase recognition
    phase_model = PhaseRecognitionNet(num_phases=12, use_pretrained=True).to(device)
    phase_model.load_state_dict(torch.load(args.phase_model, map_location=device), strict=True)
    phase_model.eval()

    # 3) VLM + Feedback
    vlm = VLMModel("google/vit-base-patch16-224","distilbert-base-uncased").to(device)
    feedback_gen = FeedbackGenerator(vlm, device=device)

    # 4) Heuristic technique
    technique_advisor = TechniqueAdvisor()

    # Albumentations transforms
    seg_transform = A.Compose([
        A.Resize(512,512),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    vlm_transform = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    phase_transform = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Failed to open:", args.video)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video: {args.video}, frames={total_frames}, fps={fps:.2f}")

    cv2.namedWindow("SurgeryFeedback", cv2.WINDOW_NORMAL)
    if total_frames>0:
        cv2.createTrackbar("Frame", "SurgeryFeedback", 0, total_frames-1, nothing)

    playing = False

    # Store feedback: (frame_idx, time_sec, technique_feedback, vlm_feedback)
    feedback_log = []
    last_frame_idx = -1
    last_tech = ""
    last_vlm = ""

    while True:
        if playing and total_frames>0:
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            if total_frames>0:
                tbpos = cv2.getTrackbarPos("Frame", "SurgeryFeedback")
                cap.set(cv2.CAP_PROP_POS_FRAMES, tbpos)
                frame_idx = tbpos

        ret, frame_bgr = cap.read()
        if not ret:
            if total_frames>0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                print("End of video or read error.")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        time_sec = frame_idx / fps

        # Skip processing if it's the same frame we just did (avoid duplication)
        if frame_idx == last_frame_idx:
            # show the same image
            overlay = frame_bgr.copy()
            cv2.putText(overlay, "Press 'p' to Play/Pause, 'q' to Quit", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
            cv2.putText(overlay, "Press 'p' to Play/Pause, 'q' to Quit", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
            cv2.imshow("SurgeryFeedback", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting.")
                break
            elif key == ord('p') or key==32:
                playing = not playing
                print(f"Playing={playing}")
            continue

        last_frame_idx = frame_idx

        # (A) Segmentation
        with torch.no_grad():
            seg_in = seg_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            seg_out = seg_model(seg_in)["out"]
            seg_pred = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()

        # (B) Phase
        with torch.no_grad():
            ph_in = phase_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            ph_out = phase_model(ph_in)
            ph_label = int(torch.argmax(ph_out, dim=1).item())
        phase_name = PHASE_MAP.get(ph_label, "Unknown")

        # (C) VLM feedback
        vlm_in = vlm_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
        new_vlm_feedback = feedback_gen.get_frame_feedback(vlm_in, phase_name)

        # (D) Heuristic
        new_tech_feedback = technique_advisor.get_feedback(seg_pred, phase_name)

        # If both are the same as last time, skip storing
        if (new_tech_feedback.strip()==last_tech.strip() and
            new_vlm_feedback.strip()==last_vlm.strip()):
            # no change => skip
            pass
        else:
            # store
            feedback_log.append((frame_idx, time_sec, new_tech_feedback, new_vlm_feedback))
            last_tech = new_tech_feedback
            last_vlm = new_vlm_feedback

        # Visualization
        overlay = frame_bgr.copy()
        unique_ids = np.unique(seg_pred)
        for cid in unique_ids:
            if cid==0: continue
            mask_bin = (seg_pred==cid).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0,255,0), 2)

        cv2.putText(overlay, f"Frame={frame_idx}  Time={time_sec:.2f}s", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)
        cv2.putText(overlay, f"Frame={frame_idx}  Time={time_sec:.2f}s", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)

        cv2.putText(overlay, f"Phase: {phase_name}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
        cv2.putText(overlay, f"Phase: {phase_name}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)

        cv2.putText(overlay, "Press 'p' to Play/Pause, 'q' to Quit", (10,75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)
        cv2.putText(overlay, "Press 'p' to Play/Pause, 'q' to Quit", (10,75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)

        if playing and total_frames>0:
            cv2.setTrackbarPos("Frame", "SurgeryFeedback", frame_idx)

        cv2.imshow("SurgeryFeedback", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting early.")
            break
        elif key == ord('p') or key==32:
            playing = not playing
            print(f"Playing={playing}")

    cap.release()
    cv2.destroyAllWindows()

    # Print final summary once the entire run is done
    print("\n=== FINAL FEEDBACK SUMMARY ===")
    for (fidx, sec, ttext, vtext) in feedback_log:
        print(f"Frame={fidx}, Time={sec:.2f}s")
        if ttext.strip():
            print("  TechniqueAdvisor:", ttext.replace("\n"," | "))
        print("  VLM:", vtext)
    print("=== END SUMMARY ===")

if __name__=="__main__":
    main()