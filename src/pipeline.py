"""
src/pipeline.py
End-to-end orchestration pipeline for the AskPlayStation AI customer support agent.
Chains: Ingest -> Classify -> Retrieve -> Respond -> Route.
"""

import os
import sys
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intents import classify_intent
from src.retrieval import retrieve_similar_resolutions
from src.responder import generate_grounded_reply
from src.router import route_message

class SupportPipeline:
    """
    Main orchestration class for AskPlayStation AI Support Agent.
    """

    def __init__(self):
        pass

    def run(self, customer_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a customer message end-to-end:
        1. Classify intent
        2. Retrieve historical resolution precedents
        3. Draft grounded response
        4. Decide AUTO_HANDLE vs ESCALATE with inspectable reasoning
        """
        # Step 1: Classify Intent
        classification = classify_intent(customer_text, context=context)
        intent = classification.get("intent", "general_inquiry_other")
        confidence = classification.get("confidence", 0.85)

        # Step 2: Retrieve Historical Resolutions for Grounding
        retrieved = retrieve_similar_resolutions(customer_text, top_k=3)

        # Step 3: Draft Grounded Response
        reply_obj = generate_grounded_reply(
            customer_text=customer_text,
            intent=intent,
            retrieved_resolutions=retrieved,
            context=context
        )
        grounded_reply = reply_obj.get("reply", "")

        # Step 4: Decide Routing (AUTO_HANDLE vs ESCALATE)
        routing = route_message(
            customer_text=customer_text,
            intent=intent,
            confidence=confidence,
            context=context
        )
        decision = routing.get("decision", "AUTO_HANDLE")
        reason = routing.get("reason", "")

        return {
            "customer_text": customer_text,
            "intent": intent,
            "confidence": confidence,
            "grounded_reply": grounded_reply,
            "decision": decision,
            "reason": reason,
            "retrieved_precedents": retrieved,
            "reply_source": reply_obj.get("source", "unknown"),
            "policy_rule_triggered": routing.get("policy_rule_triggered")
        }

# Global singleton pipeline
_PIPELINE_INSTANCE: Optional[SupportPipeline] = None

def get_pipeline() -> SupportPipeline:
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = SupportPipeline()
    return _PIPELINE_INSTANCE

def run_pipeline(customer_text: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Convenience functional interface for CLI and testing."""
    pipeline = get_pipeline()
    return pipeline.run(customer_text, context)

if __name__ == "__main__":
    sample = "my psn account got banned for no reason"
    res = run_pipeline(sample)
    print("--- Pipeline Execution Test ---")
    print(f"Customer Text:  {res['customer_text']}")
    print(f"Intent:         {res['intent']} (confidence: {res['confidence']:.2f})")
    print(f"Grounded Reply: {res['grounded_reply']}")
    print(f"Decision:       {res['decision']}")
    print(f"Reason:         {res['reason']}")
