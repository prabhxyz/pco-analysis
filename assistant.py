class SurgeryAssistant:
    """
    Stores current context, phases, instruments, PCO risk, etc.
    Answers basic queries about phase, PCO, instruments.
    If advanced PCO analysis is requested, calls the LLM-based method from pco_intelligence.
    """
    def __init__(self, pco_intelligence=None):
        self.current_phase = "Unknown"
        self.current_pco_risk = "Unknown"
        self.current_instruments = []
        self.phase_history = []
        # pco_intelligence is an instance of PCOIntelligence
        self.pco_intelligence = pco_intelligence

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
            if "analysis" in qlower or "explain" in qlower:
                # On-demand LLM approach
                if self.pco_intelligence:
                    return self.pco_intelligence.llm_analysis(self.phase_history, self.current_instruments)
                else:
                    return f"PCO risk is {self.current_pco_risk}. LLM analysis not available."
            else:
                # Quick snapshot
                return f"PCO risk is {self.current_pco_risk}."
        elif "instrument" in qlower or "usage" in qlower:
            return f"Detected instruments: {self.current_instruments}."
        elif "history" in qlower:
            return f"Phases recognized: {self.phase_history}."
        else:
            return ("Query not recognized. Try 'phase', 'risk', or 'analysis' "
                    "for more info on PCO risk factors.")