import argparse
import cv2
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import LightweightSegModel, PhaseRecognitionNet
from technique_assessment import TechniqueAdvisor
from pco_assessment import PCORiskAssessor
from assistant import SurgeryAssistant

def get_instrument_colors():
    # BGR color mapping for 13 classes
    return [
        (0,0,0),      # 0 background
        (255,0,0),    # 1 Iris
        (0,255,0),    # 2 Pupil
        (0,0,255),    # 3 IOL
        (255,255,0),  # 4 SlitKnife
        (255,0,255),  # 5 Gauge
        (0,255,255),  # 6 Spatula
        (128,0,0),    # 7 CapsulorhexisCystotome
        (0,128,0),    # 8 PhacoTip
        (0,0,128),    # 9 I/A
        (128,128,0),  # 10 LensInjector
        (128,0,128),  # 11 CapsulorhexisForceps
        (0,128,128)   # 12 KatanaForceps
    ]

def overlay_seg(frame_rgb, seg_mask, colors, alpha=0.5):
    out = frame_rgb.copy()
    for cid in np.unique(seg_mask):
        if cid == 0:
            continue
        b,g,r = colors[cid]
        idx = (seg_mask == cid)
        out[idx, 0] = (1 - alpha)*out[idx, 0] + alpha*r
        out[idx, 1] = (1 - alpha)*out[idx, 1] + alpha*g
        out[idx, 2] = (1 - alpha)*out[idx, 2] + alpha*b
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Path to .mp4 or camera index (default=0).")
    parser.add_argument("--seg_model", type=str, default="lightweight_seg.pth", help="Path to the saved segmentation model")
    parser.add_argument("--phase_model", type=str, default="phase_recognition.pth", help="Path to the saved phase model")
    parser.add_argument("--chat", action="store_true", help="Enable interactive chat in terminal.")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # 1) Load segmentation model
    #    If your saved model had aux_loss=True, let's match that.
    #    If not sure, try aux_loss=True with strict=False (below).
    seg_model = LightweightSegModel(num_classes=13, use_pretrained=False, aux_loss=True).to(device)
    # If your model was trained with aux_loss=False, do -> aux_loss=False
    # but then load with strict=False to ignore the aux keys:
    # seg_model = LightweightSegModel(num_classes=13, use_pretrained=False, aux_loss=False).to(device)

    # Load the weights
    loaded_state = torch.load(args.seg_model, map_location=device)
    # If you see "Unexpected key(s)" about aux_classifier, do strict=False
    seg_model.load_state_dict(loaded_state, strict=False)

    seg_model.eval()

    # 2) Load phase model
    #    We assume your phase model has 12 classes. Adjust if needed.
    phase_model = PhaseRecognitionNet(num_phases=12, use_pretrained=True).to(device)
    phase_model.load_state_dict(torch.load(args.phase_model, map_location=device), strict=True)
    phase_model.eval()

    # 3) Initialize technique + PCO assessors, and chat
    technique_advisor = TechniqueAdvisor()
    pco_assessor = PCORiskAssessor()
    assistant = SurgeryAssistant()

    # 4) Albumentations transforms
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

    # 5) Open video
    source = int(args.video) if (args.video is not None and args.video.isdigit()) else args.video
    cap = cv2.VideoCapture(0 if source is None else source)
    if not cap.isOpened():
        print(f"Failed to open video/camera: {source}")
        return

    instrument_colors = get_instrument_colors()
    recognized_phase_history = []

    playing = True
    print("Starting Real-Time. Press 'q' to quit, 'p' to pause.")
    if args.chat:
        print("Press 'c' to enter chat mode in terminal.")

    while True:
        if playing:
            ret, frame_bgr = cap.read()
            if not ret:
                print("End of video or read error.")
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # A) Segmentation
            seg_in = seg_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                seg_out = seg_model(seg_in)  # returns dict {'out':..., 'aux':...} if aux_loss=True
            seg_logits = seg_out["out"]   # shape [B,13,H,W]
            seg_pred = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
            overlayed = overlay_seg(frame_rgb, seg_pred, instrument_colors, alpha=0.5)

            # B) Phase recognition
            phase_in = phase_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                phase_logits = phase_model(phase_in)
            phase_label = int(torch.argmax(phase_logits, dim=1).item())

            # dummy mapping for your 12 phases
            phase_map = {
                0:"Incision", 1:"Viscoelastic", 2:"Capsulorhexis", 3:"Hydrodissection",
                4:"Phacoemulsification", 5:"Irrigation/Aspiration", 6:"Capsule Polishing",
                7:"Lens Implantation", 8:"Lens positioning", 9:"Viscoelastic Suction",
                10:"Tonifying/Antibiotics", 11:"Other"
            }
            phase_name = phase_map.get(phase_label, "Unknown")
            recognized_phase_history.append(phase_name)

            # C) Technique feedback
            technique_text = technique_advisor.get_feedback(seg_pred, phase_name)

            # D) PCO risk
            pco_risk = pco_assessor.estimate_risk(seg_pred, recognized_phase_history)

            # E) Update chat assistant
            present_ids = np.unique(seg_pred)
            instr_names = []
            for cid in present_ids:
                if cid == 0: continue
                instr_names.append(technique_advisor.INSTRUMENT_NAMES[cid])
            assistant.update_context(phase_name, pco_risk, instr_names)

            # F) Visualize
            overlay_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
            cv2.putText(overlay_bgr, f"Phase: {phase_name}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)
            cv2.putText(overlay_bgr, f"Phase: {phase_name}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)

            cv2.putText(overlay_bgr, f"PCO Risk: {pco_risk}", (10,45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)
            cv2.putText(overlay_bgr, f"PCO Risk: {pco_risk}", (10,45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),1)

            # Show technique feedback (up to 3 lines)
            lines = technique_text.split("\n")
            ystart = 70
            for i, ln in enumerate(lines[:3]):
                cv2.putText(overlay_bgr, ln, (10, ystart),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                cv2.putText(overlay_bgr, ln, (10, ystart),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                ystart += 20

            cv2.imshow("CataractSurgeryGuidance", overlay_bgr)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('p'):
            playing = not playing
        elif key == ord('c') and args.chat:
            # Terminal chat
            user_txt = input("Ask about the surgery (type 'done' to exit chat): ")
            while user_txt.lower() != "done":
                ans = assistant.answer(user_txt)
                print("[Assistant]:", ans)
                user_txt = input("> ")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
