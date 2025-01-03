"""
Provides a minimal text-based assistant that answers simple questions.
Stores current phase, PCO risk, and instruments used, updating context each frame.
"""

class SurgeryAssistant:
    def __init__(self):
        self.current_phase = "Unknown"
        self.current_pco_risk = "Unknown"
        self.current_instruments = []
        self.phase_history = []

    def update_context(self, phase_name, pco_risk, instruments):
        self.current_phase = phase_name
        self.current_pco_risk = pco_risk
        self.current_instruments = instruments
        if phase_name not in self.phase_history:
            self.phase_history.append(phase_name)

    def answer(self, user_query):
        qlower = user_query.lower()
        if "phase" in qlower:
            return f"Current recognized phase is {self.current_phase}."
        elif "pco" in qlower or "risk" in qlower:
            return f"Estimated PCO risk is {self.current_pco_risk}."
        elif "instrument" in qlower or "usage" in qlower:
            return f"Detected instruments: {self.current_instruments}."
        elif "history" in qlower:
            return f"Phases recognized so far: {self.phase_history}."
        else:
            return "No direct answer found. Query can reference 'phase', 'risk', or 'instruments'."