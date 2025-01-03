import cv2
import numpy as np

class FlowTracker:
    """
    Demonstrates a simplified approach to track instrument centroids with optical flow.
    If RAFT is used, it would be more complex. This example uses cv2.calcOpticalFlowFarneback.
    """
    def __init__(self):
        self.prev_gray = None
        self.prev_points = None

    def track_instruments(self, frame_gray, centroids):
        """
        centroids: list of (cx, cy) for each instrument region.
        If there's a previous frame, tries to estimate their new positions with optical flow.
        """
        if self.prev_gray is None:
            # first frame
            self.prev_gray = frame_gray
            self.prev_points = np.array(centroids, dtype=np.float32).reshape(-1,1,2)
            return centroids

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, frame_gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # This is a dense flow, a real RAFT approach would be more advanced.

        # a naive approach might average flow around each centroid
        new_positions = []
        for (cx, cy) in centroids:
            cxi = int(cx)
            cyi = int(cy)
            # clamp
            if cxi<0 or cyi<0 or cxi>=flow.shape[1] or cyi>=flow.shape[0]:
                new_positions.append((cx, cy))
                continue
            flow_vec = flow[cyi, cxi]  # (dx, dy)
            new_cx = cx + flow_vec[0]
            new_cy = cy + flow_vec[1]
            new_positions.append((new_cx, new_cy))

        self.prev_gray = frame_gray
        return new_positions
