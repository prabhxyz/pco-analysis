import numpy as np
import math
from collections import deque

class Tracklet:
    """
    Holds state for a single tracked object (instrument).
    """
    def __init__(self, track_id, bbox, conf=1.0):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.conf = conf
        self.misses = 0
        self.history = deque(maxlen=30)

    def update(self, new_bbox):
        self.bbox = new_bbox
        self.misses = 0
        self.history.append(new_bbox)

class ByteTracker:
    """
    A minimal bounding-box tracker that tries to link detections across frames.
    This is not the official ByteTrack but a demonstration of principle.
    """
    def __init__(self, max_distance=100, max_misses=5):
        self.max_distance = max_distance
        self.max_misses = max_misses
        self.next_id = 1
        self.active_tracks = []

    def distance(self, bboxA, bboxB):
        # center distance
        cxA = (bboxA[0] + bboxA[2]) / 2
        cyA = (bboxA[1] + bboxA[3]) / 2
        cxB = (bboxB[0] + bboxB[2]) / 2
        cyB = (bboxB[1] + bboxB[3]) / 2
        return math.hypot(cxA - cxB, cyA - cyB)

    def update_tracks(self, bboxes):
        """
        bboxes: list of bounding boxes [x1,y1,x2,y2]
        """
        # Convert to arrays
        bboxes_array = np.array(bboxes, dtype=float)

        # Step 1: Attempt to match existing tracklets
        matched_indices = set()
        for track in self.active_tracks:
            track.misses += 1
            best_dist = self.max_distance
            best_idx = -1
            for i, bb in enumerate(bboxes_array):
                if i in matched_indices:
                    continue
                dist = self.distance(track.bbox, bb)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx >= 0 and best_dist < self.max_distance:
                track.update(bboxes_array[best_idx])
                matched_indices.add(best_idx)
            # else track remains unmatched, misses increment

        # Step 2: spawn new tracks for unmatched detections
        for i, bb in enumerate(bboxes_array):
            if i not in matched_indices:
                new_track = Tracklet(self.next_id, bb, conf=1.0)
                self.next_id += 1
                self.active_tracks.append(new_track)

        # Step 3: remove dead tracks
        self.active_tracks = [
            t for t in self.active_tracks if t.misses <= self.max_misses
        ]

        # Return current tracks
        return self.active_tracks