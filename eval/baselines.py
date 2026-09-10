"""
eval/baselines.py
Defines the two comparison baselines for evaluation:
1. Trivial Baseline: Majority-class intent prediction + templated generic reply.
2. Simple Baseline: Keyword-rule classifier + retrieval-only reply (no LLM generation).
"""

import os
import sys
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intents import classify_intent_heuristic
from src.retrieval import retrieve_similar_resolutions

class TrivialBaseline:
    """
    Baseline 1 (Trivial):
    - Always predicts the majority class intent in natural Twitter data ('general_inquiry_other').
    - Returns a static canned generic reply.
    - Always AUTO_HANDLE.
    """
    def __init__(self):
        self.majority_intent = "general_inquiry_other"
        self.canned_reply = (
            "Hi there! Thanks for reaching out to PlayStation Support. "
            "Please visit https://www.playstation.com/support for assistance with your account and console."
        )

    def run(self, customer_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "confidence": 0.50,
            "grounded_reply": self.canned_reply,
            "decision": "AUTO_HANDLE",
            "reason": "Trivial baseline default: majority-class generic automated handling.",
            "source": "trivial_baseline"
        }

class SimpleBaseline:
    """
    Baseline 2 (Simple):
    - Classifies intent using deterministic keyword rules (no LLM).
    - Retrieves top-1 historical resolution and directly uses the raw historical tweet reply (no LLM generation).
    - Uses naive keyword safety router for escalation.
    """
    def __init__(self):
        pass

    def run(self, customer_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        # 1. Keyword classification
        classification = classify_intent_heuristic(customer_text)
        intent = classification["intent"]
        confidence = classification["confidence"]

        # 2. Retrieval-only response (no LLM generation/rewriting)
        retrieved = retrieve_similar_resolutions(customer_text, top_k=1)
        if retrieved:
            raw_past_reply = retrieved[0]["historical_reply"]
        else:
            raw_past_reply = "Hello! Please check our support page at playstation.com for help."

        # 3. Naive keyword escalation rules
        text_lower = customer_text.lower()
        if any(k in text_lower for k in ["ban", "banned", "suspension", "refund", "charged twice", "hacked"]):
            decision = "ESCALATE"
            reason = "Simple keyword match on sensitive escalation terms."
        else:
            decision = "AUTO_HANDLE"
            reason = "No sensitive escalation keywords detected."

        return {
            "intent": intent,
            "confidence": confidence,
            "grounded_reply": raw_past_reply,
            "decision": decision,
            "reason": reason,
            "source": "simple_baseline",
            "retrieved_precedent": retrieved[0] if retrieved else None
        }

if __name__ == "__main__":
    t_base = TrivialBaseline()
    s_base = SimpleBaseline()

    test_msg = "my psn account got banned for no reason"
    print("Query:", test_msg)
    print("\n--- Trivial Baseline ---")
    print(t_base.run(test_msg))
    print("\n--- Simple Baseline ---")
    print(s_base.run(test_msg))
