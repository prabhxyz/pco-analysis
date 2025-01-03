import os
import cv2
import csv
import torch
import numpy as np
from torch.utils.data import Dataset

class CataractPhaseDataset(Dataset):
    """
    Dataset for phase recognition that parses CSV files with start/end frames for each phase.
    """
    def __init__(self, root_dir, transform=None, frame_skip=10):
        super().__init__()
        self.root_dir = root_dir
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.videos_dir = os.path.join(root_dir, "video")
        self.transform = transform
        self.frame_skip = frame_skip

        self.phase_map = {}
        self.all_videos = set()
        self.samples = []

        self._scan_videos()
        self._scan_annotations()
        self._build_samples()

        all_phase_names = sorted(list({s[2] for s in self.samples}))
        self.phase_label_map = {ph: i for i, ph in enumerate(all_phase_names)}

    def _scan_videos(self):
        """
        Scans the videos folder for case_XXXX.mp4 and stores case IDs in all_videos.
        """
        if not os.path.isdir(self.videos_dir):
            return
        for file in os.listdir(self.videos_dir):
            if file.endswith(".mp4"):
                base = file.replace(".mp4","")  # e.g. "case_4687"
                if base.startswith("case_"):
                    case_str = base[len("case_"):]
                    if case_str.isdigit():
                        self.all_videos.add(case_str)

    def _scan_annotations(self):
        """
        Recursively walks 'annotations/' to find 'case_XXXX_annotations_phases.csv' files.
        """
        for root, dirs, files in os.walk(self.annotations_dir):
            for file in files:
                if file.endswith("_annotations_phases.csv"):
                    base_no_ext = file.replace("_annotations_phases.csv", "")  # e.g. "case_4687"
                    if base_no_ext.startswith("case_"):
                        case_id_str = base_no_ext[len("case_"):]
                        if case_id_str.isdigit():
                            csv_path = os.path.join(root, file)
                            self._read_phase_csv(case_id_str, csv_path)

    def _read_phase_csv(self, case_id_str, csv_path):
        """
        Reads lines from CSV and stores segments in self.phase_map[case_id_str].
        """
        if case_id_str not in self.phase_map:
            self.phase_map[case_id_str] = []

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                phase_name = row["comment"]
                start_f = int(row["frame"])
                end_f = int(row["endFrame"])
                self.phase_map[case_id_str].append((start_f, end_f, phase_name))

    def _build_samples(self):
        """
        Builds (video_path, frame_idx, phase_name) samples from the segments in phase_map.
        """
        for case_id_str, segments in self.phase_map.items():
            if case_id_str not in self.all_videos:
                continue
            video_path = os.path.join(self.videos_dir, f"case_{case_id_str}.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for (start_f, end_f, ph) in segments:
                if start_f >= total_frames:
                    continue
                if end_f >= total_frames:
                    end_f = total_frames - 1
                for fidx in range(start_f, end_f+1, self.frame_skip):
                    if fidx < total_frames:
                        self.samples.append((video_path, fidx, ph))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_idx, phase_name = self.samples[idx]
        phase_label = self.phase_label_map[phase_name]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret or frame_bgr is None:
            frame_bgr = np.zeros((224,224,3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=frame_rgb)
            frame_tensor = transformed["image"]
        else:
            frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float() / 255.

        return frame_tensor, phase_label