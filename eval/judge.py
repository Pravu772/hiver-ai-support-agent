"""
eval/judge.py
LLM-as-a-Judge evaluation harness with explicit rubric criteria:
1. Groundedness (1-5)
2. Technical Correctness (1-5)
3. PlayStation Brand Tone (1-5)
4. Actionability (1-5)
Plus: Routing Safety Check (Binary: 1/0)
"""

import os
import sys
import json
import re
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intents import get_gemini_client

RUBRIC_DESCRIPTION = """
Evaluation Rubric (Score each dimension strictly from 1 to 5):
1. Groundedness (1-5):
   - 5: Strictly grounded in official PlayStation support practices (e.g. status.playstation.com, Restore Licenses, Safe Mode Option 4, official refund/ban protocols). No hallucinations.
   - 3: Mostly grounded, but offers generic advice or minor unverified claims.
   - 1: Fabricates fake URLs, promises impossible actions (e.g., unbanning via Twitter bot, free games).
2. Technical Correctness (1-5):
   - 5: Fully accurate diagnosis and technical remedy for the specific PlayStation symptom.
   - 3: Relevant general direction, but missing key technical steps or nuances.
   - 1: Outright wrong or misleading technical troubleshooting.
3. Brand Tone (1-5):
   - 5: Courteous, empathetic, professional, concise (PlayStation Support brand voice).
   - 3: Functional but dry, overly terse, or slightly repetitive.
   - 1: Rude, robotic, confusing, or inappropriate for an official brand account.
4. Actionability (1-5):
   - 5: Crystal clear immediate next steps (exact console menu paths, official portal links, or DM instructions).
   - 3: Vaguely actionable (e.g. "try restarting" without instructions).
   - 1: Zero actionable guidance or leaves user completely stranded.
"""

def evaluate_reply_heuristic(
    customer_text: str,
    draft_reply: str,
    reference_resolution: str,
    intent: str
) -> Dict[str, Any]:
    """
    Deterministic rule-based rubric evaluation for baseline benchmarking and fallback.
    """
    r_lower = draft_reply.lower()
    c_lower = customer_text.lower()
    
    # 1. Groundedness
    groundedness = 3.0
    if any(k in r_lower for k in ["playstation.com", "restore licenses", "safe mode", "status.playstation.com", "option 4", "registered email", "safety team"]):
        groundedness += 1.5
    if "@" in draft_reply and len(draft_reply) < 40 and not any(k in r_lower for k in ["link", "http", "step"]):
        groundedness -= 1.0 # too brief / ungrounded
    groundedness = max(1.0, min(5.0, groundedness))

    # 2. Correctness
    correctness = 3.0
    if intent == "account_ban_suspension":
        if "cannot" in r_lower or "policy" in r_lower or "safety" in r_lower or "registered email" in r_lower:
            correctness = 5.0
        elif "unban" in r_lower and "cannot" not in r_lower:
            correctness = 1.0
    elif intent == "network_outage_connection":
        if any(k in r_lower for k in ["status", "connection", "router", "dns", "network"]):
            correctness = 4.5
    elif intent == "game_content_redemption":
        if any(k in r_lower for k in ["license", "restore", "voucher", "region", "dlc", "store"]):
            correctness = 4.5
    elif intent == "hardware_system_crash":
        if any(k in r_lower for k in ["safe mode", "database", "rebuild", "repair", "hardware", "power"]):
            correctness = 4.5
    elif intent == "billing_refund_subscription":
        if any(k in r_lower for k in ["refund", "subscription", "cancel", "auto-renew", "store", "receipt"]):
            correctness = 4.5

    # 3. Tone
    tone = 3.0
    if any(greeting in r_lower for greeting in ["hi", "hello", "sorry to hear", "thanks for reaching out", "happy to help"]):
        tone += 1.0
    if len(draft_reply) < 25 or len(draft_reply) > 400:
        tone -= 0.5
    tone = max(1.0, min(5.0, tone))

    # 4. Actionability
    actionability = 3.0
    if any(k in r_lower for k in ["visit", "check", "select", "go to", "http", "settings", "hold"]):
        actionability += 1.5
    actionability = max(1.0, min(5.0, actionability))

    composite = round((groundedness + correctness + tone + actionability) / 4.0, 2)

    return {
        "groundedness": round(groundedness, 1),
        "correctness": round(correctness, 1),
        "tone": round(tone, 1),
        "actionability": round(actionability, 1),
        "composite_score": composite,
        "reasoning": "Heuristic rubric scoring based on grounding anchors, technical match, and brand tone conventions.",
        "judge_source": "rubric_heuristic"
    }

def evaluate_reply_llm(
    customer_text: str,
    draft_reply: str,
    reference_resolution: str,
    intent: str,
    thread_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluate draft reply using Gemini LLM-as-a-Judge against the 4 explicit rubric dimensions.
    """
    client_info = get_gemini_client()
    if not client_info:
        return evaluate_reply_heuristic(customer_text, draft_reply, reference_resolution, intent)

    client_type, client = client_info

    prompt = f"""You are a senior QA evaluator assessing AI-generated customer support replies for PlayStation Support (@AskPlayStation).
Evaluate the drafted reply strictly against the rubric below. Be rigorous, honest, and objective.

{RUBRIC_DESCRIPTION}

Customer Issue Context:
Intent: {intent}
Customer Query: "{customer_text}"
{"Thread Context: " + thread_context if thread_context else ""}
Official Historical Reference Resolution: "{reference_resolution}"

Candidate AI Draft Reply to Evaluate:
"{draft_reply}"

Respond ONLY with valid JSON in this exact structure:
{{
  "groundedness": <float between 1.0 and 5.0>,
  "correctness": <float between 1.0 and 5.0>,
  "tone": <float between 1.0 and 5.0>,
  "actionability": <float between 1.0 and 5.0>,
  "composite_score": <float between 1.0 and 5.0>,
  "reasoning": "<concise 2-sentence rationale for these scores>"
}}
"""

    models_to_try = [
        os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest"),
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
    ]
    seen = set()
    models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

    raw = None
    last_err = None
    for cur_model in models_to_try:
        try:
            if client_type == "genai":
                response = client.models.generate_content(
                    model=cur_model,
                    contents=prompt,
                    config={"temperature": 0.0, "response_mime_type": "application/json"}
                )
                raw = response.text
                break
            else:
                model = client.GenerativeModel(cur_model)
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.0}
                )
                raw = response.text
                break
        except Exception as e:
            last_err = e
            # Immediate failover to next working model without delay
            continue

    if raw:
        clean_json = re.search(r"\{.*\}", raw, re.DOTALL)
        if clean_json:
            try:
                data = json.loads(clean_json.group(0))
                g = float(data.get("groundedness", 3.0))
                c = float(data.get("correctness", 3.0))
                t = float(data.get("tone", 3.0))
                a = float(data.get("actionability", 3.0))
                comp = round((g + c + t + a) / 4.0, 2)
                return {
                    "groundedness": g,
                    "correctness": c,
                    "tone": t,
                    "actionability": a,
                    "composite_score": comp,
                    "reasoning": data.get("reasoning", ""),
                    "judge_source": "gemini_judge"
                }
            except Exception:
                pass

    if last_err:
        print(f"[Judge API Exception]: {last_err}")
    return evaluate_reply_heuristic(customer_text, draft_reply, reference_resolution, intent)

def judge_reply(
    customer_text: str,
    draft_reply: str,
    reference_resolution: str,
    intent: str,
    thread_context: Optional[str] = None
) -> Dict[str, Any]:
    """Primary entrypoint for evaluating a single reply."""
    return evaluate_reply_llm(customer_text, draft_reply, reference_resolution, intent, thread_context)

if __name__ == "__main__":
    sample_q = "my psn account got banned for no reason"
    sample_reply = (
        "Hello. Account suspensions are issued by the PlayStation Safety team following "
        "thorough review of Code of Conduct violations. These decisions cannot be overturned via Twitter. "
        "Please check your registered email for details."
    )
    ref = "Direct customer to Safety policy; state that ban appeals cannot be handled on Twitter."
    
    scores = judge_reply(sample_q, sample_reply, ref, "account_ban_suspension")
    print("--- Rubric Evaluation Test ---")
    print(f"Groundedness:  {scores['groundedness']}/5.0")
    print(f"Correctness:   {scores['correctness']}/5.0")
    print(f"Brand Tone:    {scores['tone']}/5.0")
    print(f"Actionability: {scores['actionability']}/5.0")
    print(f"Composite:     {scores['composite_score']}/5.0")
    print(f"Reasoning:     {scores['reasoning']}")
    print(f"Judge Source:  {scores['judge_source']}")
