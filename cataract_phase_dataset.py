import os
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CataractPhaseDataset(Dataset):
    """
    Loads phase annotations from subfolders like:
        phase/annotations/case_XXXX/case_XXXX_annotations_phases.csv
    and matches them to videos in:
        phase/video/case_XXXX.mp4

    We sample frames (startFrame..endFrame) for each annotated phase segment,
    at a user-specified 'frame_skip' interval.
    """

    def __init__(self, root_dir, transform=None, frame_skip=10):
        """
        root_dir => e.g. "datasets/Cataract-1k/phase"
        Inside root_dir:
          ├─ annotations/case_XXXX/case_XXXX_annotations_phases.csv
          └─ video/case_XXXX.mp4

        transform => Albumentations or similar, expecting transform(image=...) -> { "image": tensor }
        frame_skip => how many frames to skip between startFrame..endFrame
        """
        super().__init__()
        self.root_dir = root_dir
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.videos_dir = os.path.join(root_dir, "video")
        self.transform = transform
        self.frame_skip = frame_skip

        # This will store segments: phase_map[caseId] = [ (startF, endF, phaseName), ... ]
        self.phase_map = {}
        # We'll gather all "case_XXXX.mp4" from videos
        self.all_videos = set()
        # Final list of samples: (video_path, frame_idx, phaseName)
        self.samples = []

        self._scan_videos()         # fill self.all_videos
        self._scan_annotations()    # fill self.phase_map
        self._build_samples()       # fill self.samples

        # Build a consistent label map of distinct phase names
        all_phase_names = sorted(list({s[2] for s in self.samples}))
        self.phase_label_map = {ph: i for i, ph in enumerate(all_phase_names)}

    def _scan_videos(self):
        """
        Collect all videos in 'video/' so we know which case IDs exist.
        E.g. "case_4687.mp4" => case_id = "4687"
        """
        if not os.path.isdir(self.videos_dir):
            return
        for file in os.listdir(self.videos_dir):
            if file.endswith(".mp4"):
                base = file.replace(".mp4", "")  # e.g. "case_4687"
                # parse out the numeric portion
                if base.startswith("case_"):
                    case_id_str = base[len("case_"):]  # "4687"
                    if case_id_str.isdigit():
                        self.all_videos.add(case_id_str)

    def _scan_annotations(self):
        """
        Recursively walk 'annotations/' subfolders to find files named 'case_XXXX_annotations_phases.csv'.
        For each found file, parse the CSV to build the phase segments for that case ID.
        """
        for root, dirs, files in os.walk(self.annotations_dir):
            for file in files:
                # e.g. 'case_4687_annotations_phases.csv'
                if file.endswith("_annotations_phases.csv"):
                    # parse out the numeric ID
                    base_no_ext = file.replace("_annotations_phases.csv", "")  # e.g. "case_4687"
                    if base_no_ext.startswith("case_"):
                        case_id_str = base_no_ext[len("case_"):]  # "4687"
                        if case_id_str.isdigit():
                            csv_path = os.path.join(root, file)
                            self._read_phase_csv(case_id_str, csv_path)

    def _read_phase_csv(self, case_id_str, csv_path):
        """
        Read lines like:
          caseId,comment,frame,endFrame,sec,endSec
          4687,Incision,796,1013,13.28,16.9
          ...
        Store them in self.phase_map[case_id_str].
        """
        if case_id_str not in self.phase_map:
            self.phase_map[case_id_str] = []

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # row: { 'caseId': '4687', 'comment': 'Incision', 'frame': '796', 'endFrame': '1013', ... }
                # We assume columns 'frame' and 'endFrame' exist
                phase_name = row["comment"]
                start_f = int(row["frame"])
                end_f = int(row["endFrame"])
                # Append to list of segments
                self.phase_map[case_id_str].append((start_f, end_f, phase_name))

    def _build_samples(self):
        """
        For each case ID that has a video & phase_map segments, read total frames from the video
        and sample frames in [startFrame..endFrame].
        Then store (video_path, frame_idx, phaseName).
        """
        for case_id_str, segments in self.phase_map.items():
            # Check if we have a corresponding video
            if case_id_str not in self.all_videos:
                # no matching .mp4, skip
                continue

            video_path = os.path.join(self.videos_dir, f"case_{case_id_str}.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for (start_f, end_f, ph) in segments:
                # clamp to total_frames
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
        """
        Returns (frame_tensor, phase_label).
        """
        video_path, frame_idx, phase_name = self.samples[idx]
        phase_label = self.phase_label_map[phase_name]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()

        if not ret or frame_bgr is None:
            # fallback black frame
            frame_bgr = np.zeros((224,224,3), dtype=np.uint8)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            # Albumentations style => transform(image=frame_rgb) => { 'image': tensor }
            transformed = self.transform(image=frame_rgb)
            frame_tensor = transformed["image"]
        else:
            # minimal fallback
            frame_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float()/255.

        return frame_tensor, phase_label
