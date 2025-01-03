"""
Provides a unified system to estimate PCO risk:
  1) A lightweight "snapshot" method for real-time usage (very fast).
  2) An optional LLM-based approach for deeper analysis, but only called on demand
     to avoid lag.

Requires huggingface 'transformers' and 'accelerate' if the LLM approach is used.
"""

import numpy as np
import torch
from transformers import pipeline

class PCOIntelligence:
    def __init__(self, llm_model_name="gpt2", device="cpu"):
        """
        llm_model_name can be any HF text-generation model. The model is only used
        if llm-based analysis is requested (not every frame).
        device can be "cuda" or "cpu".
        """
        # Real-time snapshot: no initialization needed
        self.llm_enabled = False
        self.llm_predictor = None

        # Optionally load LLM
        if llm_model_name is not None:
            try:
                self.llm_predictor = pipeline(
                    "text-generation",
                    model=llm_model_name,
                    tokenizer=llm_model_name,
                    device=0 if device=="cuda" else -1
                )
                self.llm_enabled = True
                print(f"LLM loaded: {llm_model_name} on {device}. Calls only happen on demand.")
            except Exception as e:
                print("Could not load LLM pipeline:", e)
                print("Continuing without LLM approach.")
                self.llm_enabled = False

    def realtime_snapshot(self, seg_mask, phase_history):
        """
        Returns a "fast" string: "Low", "Medium", or "High", computed with minimal logic.
        Called every frame or frequently, avoiding heavy overhead.

        Example logic:
         - Checks if "Irrigation/Aspiration" is missing in phase_history => leftover lens cells
         - Checks if "Capsule Polishing" is missing => leftover lens cells
         - Checks if "Hydrodissection" is missing => incomplete removal
         - Checks if an "IntraocularLens" is recognized => partial placeholder for IOL design factor
        """
        present_ids = np.unique(seg_mask)
        risk_score = 0

        has_ia = any("Irrigation/Aspiration" in ph for ph in phase_history)
        has_polish = any("Capsule Polishing" in ph for ph in phase_history)
        if not has_ia:
            risk_score += 2
        if not has_polish:
            risk_score += 1
        if "Hydrodissection" not in phase_history:
            risk_score += 1
        if 3 not in present_ids:  # IntraocularLens
            risk_score += 1

        if risk_score >= 4:
            return "High"
        elif risk_score >= 2:
            return "Medium"
        else:
            return "Low"

    def llm_analysis(self, phase_history, instruments):
        """
        Called on demand. Summarizes phases/instruments, crafts a prompt, uses LLM to guess PCO risk.
        Does not run automatically every frame to avoid lag.
        """
        if not self.llm_enabled or self.llm_predictor is None:
            return "Cannot do LLM analysis right now (no model)."

        prompt = (
            "Context: A cataract surgery has been analyzed. "
            f"Phases recognized so far: {', '.join(phase_history)}.\n"
            f"Instruments recognized: {', '.join(instruments)}.\n"
            "Relevant PCO risk factors: incomplete cortical cleanup, missing lens polishing, leftover visco, lens design. "
            "Estimate the PCO risk in plain language (Low, Medium, or High), and explain briefly.\nAnswer: "
        )
        try:
            output = self.llm_predictor(
                prompt,
                num_return_sequences=1,
                max_new_tokens=50,
                truncation=True
            )
            text = output[0]["generated_text"]
            return text
        except Exception as e:
            return f"LLM error: {e}"
