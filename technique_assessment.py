"""
Rule-based technique checks for cataract surgery.
We attempt to detect suboptimal patterns or missing instruments for each recognized phase.
(This is purely heuristic; more advanced solutions might use a learned approach.)
"""

import numpy as np

INSTRUMENT_NAMES = [
    "Background", "Iris", "Pupil", "IntraocularLens", "SlitKnife", "Gauge", "Spatula",
    "CapsulorhexisCystotome", "PhacoTip", "IrrigationAspiration", "LensInjector",
    "CapsulorhexisForceps", "KatanaForceps"
]

# Example phases:
PHASES = [
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection", "Phacoemulsification",
    "Irrigation/Aspiration", "Capsule Polishing", "Lens Implantation", "Lens positioning",
    "Viscoelastic Suction", "Tonifying/Antibiotics"
]

class TechniqueAdvisor:
    def __init__(self):
        pass

    def get_feedback(self, seg_mask, phase_name):
        """
        seg_mask: HxW of class IDs
        phase_name: recognized phase (string)
        """
        present_ids = np.unique(seg_mask)
        present_instr = [INSTRUMENT_NAMES[i] for i in present_ids if i>0]

        tips = []
        tips.append(f"Current phase: {phase_name} | Instruments: {present_instr}")

        # Some example advanced rules:
        if phase_name == "Incision":
            if "SlitKnife" not in present_instr:
                tips.append("Warning: Incision phase but no slit knife recognized.")
            else:
                tips.append("Ensure correct incision angle to reduce wound complications.")
        if phase_name == "Capsulorhexis":
            if "CapsulorhexisCystotome" not in present_instr and "CapsulorhexisForceps" not in present_instr:
                tips.append("Cystotome or Forceps not seen. Capsulorhexis may be incomplete.")
        if phase_name == "Phacoemulsification":
            if "PhacoTip" not in present_instr:
                tips.append("No phaco tip recognized, but phaco phase detected. Check instrumentation.")
            else:
                tips.append("Stable chamber recommended; watch for corneal burn risk, adjust power/fluidics.")
        if phase_name == "Irrigation/Aspiration":
            if "IrrigationAspiration" not in present_instr:
                tips.append("I/A instrument not detected. Ensure thorough cortical cleanup.")
        if phase_name == "Lens Implantation":
            if "LensInjector" not in present_instr:
                tips.append("Lens injector missing. Confirm IOL loaded properly.")
            else:
                tips.append("Inject lens carefully; watch for orientation. Avoid capsule trauma.")

        # Add more as needed
        return "\n".join(tips)
