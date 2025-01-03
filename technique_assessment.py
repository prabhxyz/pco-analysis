import numpy as np

class TechniqueAdvisor:
    def __init__(self):
        pass

    def get_feedback(self, seg_mask, phase_name):
        unique_ids = np.unique(seg_mask)
        instruments_present = [cid for cid in unique_ids if cid!=0]

        lines = []
        lines.append(f"[Heuristic] Phase: {phase_name}, instrument IDs: {instruments_present}")

        # Example checks
        if phase_name=="Phacoemulsification" and 8 not in instruments_present:
            lines.append("No phaco tip detected in phaco phase.")
        if phase_name=="Irrigation/Aspiration" and 9 not in instruments_present:
            lines.append("I/A instrument not visible in I/A phase.")

        return "\n".join(lines)