import cv2
import numpy as np

from .byte_tracker import ByteTracker
from .flow_tracker import FlowTracker

class InstrumentTracker:
    """
    Combines bounding-box or centroid extraction from segmented polygons,
    then picks ByteTrack or flow-based approach.
    """
    def __init__(self, method="bytetrack"):
        self.method = method
        if method=="bytetrack":
            self.tracker = ByteTracker(max_distance=80)
        else:
            self.tracker = FlowTracker()
        self.last_gray = None

    def polygons_to_bboxes(self, polygons):
        """
        polygons: list of (Nx2) arrays for each instrument polygon
        returns bounding boxes [x1,y1,x2,y2]
        """
        bboxes = []
        for poly in polygons:
            x_coords = poly[:,0]
            y_coords = poly[:,1]
            x1, y1 = x_coords.min(), y_coords.min()
            x2, y2 = x_coords.max(), y_coords.max()
            bboxes.append([x1,y1,x2,y2])
        return bboxes

    def polygons_to_centroids(self, polygons):
        """
        For optical flow approach, might track centroids.
        """
        centroids = []
        for poly in polygons:
            M = cv2.moments(poly)
            if M["m00"]!=0:
                cx = M["m10"]/M["m00"]
                cy = M["m01"]/M["m00"]
                centroids.append((cx,cy))
        return centroids

    def track_instruments(self, frame_bgr, polygons):
        """
        frame_bgr: current frame
        polygons: list of polygons for each instrument from segmentation
        returns either tracked bounding boxes or new centroids
        """
        if self.method=="bytetrack":
            bboxes = self.polygons_to_bboxes(polygons)
            tracks = self.tracker.update_tracks(bboxes)
            return tracks
        else:
            # flow-based
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            centroids = self.polygons_to_centroids(polygons)
            new_positions = self.tracker.track_instruments(frame_gray, centroids)
            return new_positions
