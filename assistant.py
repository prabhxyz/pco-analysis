"""
A minimal "SurgeryAssistant" class that lets you chat in the terminal about the
current surgery. We store some contextual info (current recognized phase, PCO risk,
instruments used, etc.) and produce short answers.
"""

class SurgeryAssistant:
    def __init__(self):
        self.current_phase = "Unknown"
        self.current_pco_risk = "Unknown"
        self.current_instruments = []
        self.history_of_phases = []

    def update_context(self, phase_name, pco_risk, instruments):
        self.current_phase = phase_name
        self.current_pco_risk = pco_risk
        self.current_instruments = instruments
        if phase_name not in self.history_of_phases:
            self.history_of_phases.append(phase_name)

    def answer(self, user_query):
        """
        We do naive handling of user_query. In real usage, we might parse or
        integrate a real LLM. We'll just do rule-based replies.
        """
        qlower = user_query.lower()
        if "phase" in qlower:
            return f"The current recognized phase is {self.current_phase}."
        elif "pco" in qlower or "risk" in qlower:
            return f"Estimated PCO risk right now is {self.current_pco_risk}."
        elif "instrument" in qlower or "what's in use" in qlower:
            return f"Instruments detected: {self.current_instruments}."
        elif "history" in qlower:
            return f"The phases recognized so far: {self.history_of_phases}."
        else:
            # fallback
            return ("I'm keeping track of the phase, instruments, and PCO risk. "
                    "Ask about 'phase', 'risk', or 'instruments' for more info.")
