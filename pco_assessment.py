"""
Naive approach to PCO risk estimation based on recognized instruments and phases.
No dataset is required; relies on heuristic rules.
"""

import numpy as np

class PCORiskAssessor:
    def __init__(self):
        pass

    def estimate_risk(self, seg_mask, phase_name_history):
        """
        Checks recognized instruments and list of recognized phase names to produce "Low", "Medium", or "High".
        """
        present_ids = np.unique(seg_mask)
        risk_score = 0

        # If "Irrigation/Aspiration" never appeared in entire procedure
        has_ia = any("Irrigation/Aspiration" in ph for ph in phase_name_history)
        if not has_ia:
            risk_score += 2

        if "Lens Implantation" in phase_name_history and "Capsule Polishing" not in phase_name_history:
            risk_score += 1

        if "Viscoelastic Suction" not in phase_name_history:
            risk_score += 1

        if 3 not in present_ids:  # 3 => IntraocularLens
            risk_score += 1

        if risk_score >= 4:
            return "High"
        elif risk_score >= 2:
            return "Medium"
        else:
            return "Low"