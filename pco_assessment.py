"""
PCO risk estimation using a set of known risk factors or "markers" from the real-time
analysis, with no dedicated dataset needed.

We combine recognized instruments, recognized phases, and simple heuristics about
capsular cleanliness, usage of I/A, thorough lens polishing, etc.
"""

import numpy as np

class PCORiskAssessor:
    def __init__(self):
        pass

    def estimate_risk(self, seg_mask, phase_name_history):
        """
        seg_mask: current frame segmentation mask
        phase_name_history: a list of phases recognized over time in the surgery
            (so we can see if certain steps might have been skipped or truncated).
        Returns "Low", "Medium", or "High".
        """
        present_ids = np.unique(seg_mask)
        present_instruments = set(present_ids.tolist())

        risk_score = 0

        # If irrigation/aspiration never appeared in the entire procedure, big risk factor
        has_ia = any("Irrigation/Aspiration" in ph for ph in phase_name_history)
        if not has_ia:
            risk_score += 2

        # If "Lens Implantation" phase occurred but we rarely saw thorough "Capsule Polishing" or "Irrigation/Aspiration"
        # let's add to risk
        if "Lens Implantation" in phase_name_history and "Capsule Polishing" not in phase_name_history:
            risk_score += 1

        # If final phases are missing "Viscoelastic Suction", leftover visco might hamper lens clarity
        if "Viscoelastic Suction" not in phase_name_history:
            risk_score += 1

        # If no lens was recognized at all
        if 3 not in present_instruments:  # 3 => IntraocularLens
            risk_score += 1

        if risk_score >= 4:
            return "High"
        elif risk_score >= 2:
            return "Medium"
        else:
            return "Low"