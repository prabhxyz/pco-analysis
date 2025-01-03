"""
Rule-based technique assessment. Checks recognized phase and instrument IDs
to provide short feedback or warnings.
"""

import numpy as np

INSTRUMENT_NAMES = [
    "Background",
    "Iris",
    "Pupil",
    "IntraocularLens",
    "SlitKnife",
    "Gauge",
    "Spatula",
    "CapsulorhexisCystotome",
    "PhacoTip",
    "IrrigationAspiration",
    "LensInjector",
    "CapsulorhexisForceps",
    "KatanaForceps"
]

PHASE_NAMES = [
    "Incision",
    "Viscoelastic",
    "Capsulorhexis",
    "Hydrodissection",
    "Phacoemulsification",
    "Irrigation/Aspiration",
    "Capsule Polishing",
    "Lens Implantation",
    "Lens positioning",
    "Viscoelastic Suction",
    "Tonifying/Antibiotics",
    "Other"
]

class TechniqueAdvisor:
    def __init__(self):
        pass

    def get_feedback(self, seg_mask, phase_name):
        """
        Returns a short string with technique warnings or suggestions.
        """
        present_ids = np.unique(seg_mask)
        present_instr = [INSTRUMENT_NAMES[i] for i in present_ids if i != 0]

        lines = []
        lines.append(f"Phase: {phase_name}, instruments: {present_instr}")

        if phase_name == "Phacoemulsification" and "PhacoTip" not in present_instr:
            lines.append("No phaco tip detected in phaco phase.")

        if phase_name == "Incision" and "SlitKnife" not in present_instr:
            lines.append("Incision phase, but no slit knife recognized.")

        return "\n".join(lines)