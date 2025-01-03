import argparse
import cv2
import torch
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from training.models import InstrumentSegModel, PhaseRecognitionNet
from instrument_tracking.instrument_tracker import InstrumentTracker
from pco_risk_module import PCORiskModule
from technique_assessment import TechniqueAdvisor

def polygons_from_mask(seg_mask):
    """
    Extract polygons from a segmentation mask (like cv2.findContours).
    Each instrument class > 0 => findContours => store polygon.
    Returns dict: {class_id: [list_of_polygons], ...}
    """
    polygons_dict = {}
    unique_ids = np.unique(seg_mask)
    for cid in unique_ids:
        if cid == 0:  # background
            continue
        # isolate that class
        mask_bin = (seg_mask==cid).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons_dict.setdefault(cid, []).extend(contours)
    return polygons_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--seg_model", type=str, default="instrument_seg_model.pth")
    parser.add_argument("--phase_model", type=str, default="phase_recognition.pth")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Load models
    seg_model = InstrumentSegModel(num_classes=13).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model, map_location=device), strict=False)
    seg_model.eval()

    phase_model = PhaseRecognitionNet(num_phases=12).to(device)
    phase_model.load_state_dict(torch.load(args.phase_model, map_location=device), strict=True)
    phase_model.eval()

    technique_advisor = TechniqueAdvisor()
    pco_module = PCORiskModule()

    # Create an instrument tracker (ByteTrack or flow-based).
    # Let's pick "bytetrack" for bounding boxes:
    tracker = InstrumentTracker(method="bytetrack")

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

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print("Failed to open video/camera.")
        return

    recognized_phases = []
    # Keep a dictionary of usage frames for certain instruments
    usage_frames = {
        "IrrigationAspiration": 0,
        "CapsulePolishing": 0,
        # can track others
    }

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 1) Segmentation
        with torch.no_grad():
            seg_in = seg_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            seg_out = seg_model(seg_in)["out"]
            seg_pred = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()

        # Extract polygons per class
        polygons_dict = polygons_from_mask(seg_pred)

        # Combine all polygons into one list for the tracker (some trackers might want them separate by class)
        all_polygons = []
        for cid, poly_list in polygons_dict.items():
            all_polygons.extend(poly_list)

        # 2) Instrument tracking
        # track_instruments returns track states if using bounding boxes
        # We'll get them as "tracks" with track_id, bounding box, etc.
        tracks = tracker.track_instruments(frame_bgr, all_polygons)

        # 3) Phase recognition
        with torch.no_grad():
            ph_in = phase_transform(image=frame_rgb)["image"].unsqueeze(0).to(device)
            phase_logits = phase_model(ph_in)
            phase_label = int(torch.argmax(phase_logits, dim=1).item())

        phase_map = {
            0:"Incision",1:"Viscoelastic",2:"Capsulorhexis",3:"Hydrodissection",
            4:"Phacoemulsification",5:"Irrigation/Aspiration",6:"Capsule Polishing",
            7:"Lens Implantation",8:"Lens positioning",9:"Viscoelastic Suction",
            10:"Tonifying/Antibiotics",11:"Other"
        }
        phase_name = phase_map.get(phase_label, "Unknown")
        if phase_name not in recognized_phases:
            recognized_phases.append(phase_name)

        # If recognized phase is I/A, increment usage_frames
        if phase_name == "Irrigation/Aspiration":
            usage_frames["IrrigationAspiration"] += 1
        if phase_name == "Capsule Polishing":
            usage_frames["CapsulePolishing"] += 1

        # 4) Optional technique feedback
        feedback = technique_advisor.get_feedback(seg_pred, phase_name)
        if feedback.strip():
            print("[TechniqueAdvisor]:", feedback)

        # 5) Visualize tracking
        # ByteTrack or bounding box-based approach returns track.bbox => [x1,y1,x2,y2].
        overlay = frame_bgr.copy()
        for t in tracks:
            x1,y1,x2,y2 = t.bbox
            color = (0,255,0)
            cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), color,2)
            cv2.putText(overlay, f"ID:{t.track_id}", (int(x1),int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 6) Show current phase
        cv2.putText(overlay, f"Phase: {phase_name}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)
        cv2.putText(overlay, f"Phase: {phase_name}", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1)

        # 7) Periodically or at end, compute PCO risk
        # A simple example: compute it every frame or every few frames
        pco_risk = pco_module.compute_risk(recognized_phases, {
            "IrrigationAspiration_frames": usage_frames["IrrigationAspiration"],
            "CapsulePolishing_frames": usage_frames["CapsulePolishing"]
        })
        cv2.putText(overlay, f"PCO risk: {pco_risk}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)
        cv2.putText(overlay, f"PCO risk: {pco_risk}", (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),1)

        cv2.imshow("PipelineDemo", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()