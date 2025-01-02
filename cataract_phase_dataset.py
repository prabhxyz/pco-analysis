import os
import cv2
import csv
import torch
import numpy as np
from torch.utils.data import Dataset

class CataractPhaseDataset(Dataset):
    """
    For the Cataract-1k 'phase' dataset structure:
      root_dir
       └── video
            └── case_XXXX.mp4
       └── annotations
            └── case_XXXX_annotations_phases.csv

    Each CSV row:  caseId,phaseName,frame,startFrame,endFrame,sec,startSec,endSec, etc.

    We'll load entire videos, but during training we'll sample frames with assigned phase labels.
    You can do a chunk-based approach (for LSTM) or a single-frame approach. Below is single-frame.
    """
    def __init__(self, root_dir, transform=None, frame_skip=10):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip
        self.samples = []  # each entry: (video_path, frame_idx, phase_label)

        video_dir = os.path.join(root_dir, "video")
        ann_dir = os.path.join(root_dir, "annotations")
        mp4s = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

        # Build a dictionary of {case_id -> phase segments}, each segment = (start_frame, end_frame, phaseName)
        self.phase_map = {}
        for file in os.listdir(ann_dir):
            if not file.endswith(".csv"):
                continue
            csv_path = os.path.join(ann_dir, file)
            case_id = file.split("_")[0]  # "4687_annotations_phases.csv" => "4687"
            self.phase_map[case_id] = []
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # row: caseId, comment(phaseName), frame, endFrame, sec, endSec
                    ph = row["comment"]  # e.g. "Incision"
                    start_f = int(row["frame"])
                    end_f = int(row["endFrame"])
                    self.phase_map[case_id].append((start_f, end_f, ph))

        # Now build the training samples. We read each MP4, get length, for each segment sample frames.
        for mp4 in mp4s:
            # e.g. "case_4687.mp4"
            case_str = mp4.replace(".mp4", "").replace("case_","")  # "4687"
            if case_str not in self.phase_map:
                continue
            video_path = os.path.join(video_dir, mp4)

            segs = self.phase_map[case_str]  # list of segments
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            # We'll retrieve total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for (start_f, end_f, ph) in segs:
                # sample frames from start_f..end_f
                for fidx in range(start_f, min(end_f+1, total_frames), self.frame_skip):
                    self.samples.append((video_path, fidx, ph))

        # Optionally define a consistent set of phase label mappings
        self.phase_label_map = {}
        all_phases = sorted(list({s[2] for s in self.samples}))
        for i, phname in enumerate(all_phases):
            self.phase_label_map[phname] = i

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_idx, phase_name = self.samples[idx]
        phase_label = self.phase_label_map[phase_name]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()
        if not ret:
            frame_bgr = np.zeros((224,224,3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Use Albumentations (image=frame_rgb)
        if self.transform:
            transformed = self.transform(image=frame_rgb)
            frame_t = transformed["image"]  # a tensor
        else:
            frame_t = torch.from_numpy(frame_rgb).permute(2,0,1).float()/255.

        return frame_t, phase_label