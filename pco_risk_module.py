import numpy as np

class PCORiskModule:
    """
    Predicts PCO risk from recognized phases + instrument tracking stats.
    This example collects:
     - total frames or time where irrigation/aspiration was seen
     - usage of capsule polishing instrument
     - whether leftover lens cells might remain
    """
    def __init__(self):
        pass

    def compute_risk(self, phase_history, tracking_stats):
        """
        phase_history: list of recognized phases over time
        tracking_stats: dictionary with info about how long certain instruments were used,
                        or how thoroughly they've been moved in the capsule region.
        Returns "Low", "Medium", or "High" (example).
        """

        risk_score = 0

        # If "Irrigation/Aspiration" was rarely used
        ia_time = tracking_stats.get("IrrigationAspiration_frames", 0)
        if ia_time < 30:  # e.g., less than 30 frames in total usage
            risk_score += 2

        # If "Capsule Polishing" instrument usage is missing or short
        polish_time = tracking_stats.get("CapsulePolishing_frames", 0)
        if polish_time < 10:
            risk_score += 1

        # If phases do not contain "Capsule Polishing"
        if not any("Capsule Polishing" in ph for ph in phase_history):
            risk_score += 1

        # If "Hydrodissection" not recognized
        if not any("Hydrodissection" in ph for ph in phase_history):
            risk_score += 1

        # Summarize
        if risk_score >= 4:
            return "High"
        elif risk_score >= 2:
            return "Medium"
        else:
            return "Low"
