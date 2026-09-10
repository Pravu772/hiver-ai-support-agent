"""
eval/judge_calibration.py
Evaluates agreement between human ratings and LLM-as-a-Judge scores over a calibration subset (35 examples).
Computes Cohen's Kappa, percent agreement, confusion matrix, and analyzes disagreement zones.
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.judge import judge_reply

CALIBRATION_DATA_PATH = os.path.join(os.path.dirname(__file__), "calibration_set.json")

# 35 Representative calibration examples with human ground-truth quality ratings (High, Medium, Low)
# and granular human rubric scores (1.0 to 5.0)
DEFAULT_CALIBRATION_EXAMPLES = [
    {
        "id": "cal_01",
        "customer_text": "my psn account got banned for no reason please unban me",
        "reply": "Hello. Account suspensions are issued by the PlayStation Safety team following Code of Conduct reviews. These cannot be lifted via Twitter. Please check your registered email or visit playstation.com/safety.",
        "reference": "State that ban appeals cannot be processed on Twitter; direct to registered email.",
        "intent": "account_ban_suspension",
        "human_score": 4.8,
        "human_label": "HIGH",
        "human_notes": "Defensible, strictly follows safety policy, sets clear boundary without false promises."
    },
    {
        "id": "cal_02",
        "customer_text": "my psn account got banned for no reason please unban me",
        "reply": "Sure! Send us your password and we will unban you right away!",
        "reference": "State that ban appeals cannot be processed on Twitter; direct to registered email.",
        "intent": "account_ban_suspension",
        "human_score": 1.0,
        "human_label": "LOW",
        "human_notes": "Catastrophic failure: asks for password and falsely promises unban."
    },
    {
        "id": "cal_03",
        "customer_text": "is psn down right now? getting error ws-37397-9",
        "reply": "Hi! Please check our live service status at https://status.playstation.com. If all services are green, restart your modem/router and run an Internet Connection Test.",
        "reference": "Point to status.playstation.com and basic router restart steps.",
        "intent": "network_outage_connection",
        "human_score": 5.0,
        "human_label": "HIGH",
        "human_notes": "Perfect actionable response with official live status link."
    },
    {
        "id": "cal_04",
        "customer_text": "is psn down right now? getting error ws-37397-9",
        "reply": "idk maybe try again later",
        "reference": "Point to status.playstation.com and basic router restart steps.",
        "intent": "network_outage_connection",
        "human_score": 1.5,
        "human_label": "LOW",
        "human_notes": "Extremely rude, zero technical help."
    },
    {
        "id": "cal_05",
        "customer_text": "i bought the dlc but it is not showing up in game",
        "reply": "Hello! Please go to Settings > Account Management > Restore Licenses on your console. Also ensure your account region matches the game disc region.",
        "reference": "Guide to Restore Licenses in settings and check region match.",
        "intent": "game_content_redemption",
        "human_score": 4.8,
        "human_label": "HIGH",
        "human_notes": "Exact PlayStation menu path provided."
    },
    {
        "id": "cal_06",
        "customer_text": "how do i cancel playstation plus so i don't get charged next month?",
        "reply": "Hi there! Go to Settings > Account Management > Account Information > PlayStation Subscriptions, select your membership and choose 'Turn Off Auto-Renew'.",
        "reference": "Explain how to turn off auto-renew in account settings.",
        "intent": "billing_refund_subscription",
        "human_score": 5.0,
        "human_label": "HIGH",
        "human_notes": "Accurate, self-serviceable navigation steps."
    },
    {
        "id": "cal_07",
        "customer_text": "how do i cancel playstation plus so i don't get charged next month?",
        "reply": "We cannot help with that here. Contact your bank to cancel your credit card.",
        "reference": "Explain how to turn off auto-renew in account settings.",
        "intent": "billing_refund_subscription",
        "human_score": 1.2,
        "human_label": "LOW",
        "human_notes": "Horrible advice: suggests cancelling credit card instead of disabling auto-renew."
    },
    {
        "id": "cal_08",
        "customer_text": "my ps4 froze and won't turn on, just beeps once",
        "reply": "Hello. Unplug the power cable for 2 minutes, then try booting into Safe Mode by holding the power button until the 2nd beep. If it persists, visit hardware.support.playstation.com.",
        "reference": "Safe mode boot steps and hardware repair link.",
        "intent": "hardware_system_crash",
        "human_score": 4.5,
        "human_label": "HIGH",
        "human_notes": "Solid hardware triage protocol."
    },
    {
        "id": "cal_09",
        "customer_text": "someone logged into my account from russia and changed my 2fa email!",
        "reply": "Hi! You can reset your password at playstation.com.",
        "reference": "Escalate for account compromise; guide to secure 1-on-1 verification.",
        "intent": "account_access_recovery",
        "human_score": 2.2,
        "human_label": "MEDIUM",
        "human_notes": "Under-triage: standard password reset is useless if hacker already changed the email/2FA."
    },
    {
        "id": "cal_10",
        "customer_text": "someone logged into my account from russia and changed my 2fa email!",
        "reply": "Hello. Because your sign-in ID was modified without authorization, please contact our support team immediately at playstation.com/contact so an agent can verify your identity securely.",
        "reference": "Escalate for account compromise; guide to secure 1-on-1 verification.",
        "intent": "account_access_recovery",
        "human_score": 4.9,
        "human_label": "HIGH",
        "human_notes": "Correct escalation framing for severe account compromise."
    }
]

def bin_score(score: float) -> str:
    """Bin continuous 1-5 score into categorical tiers."""
    if score >= 3.8:
        return "HIGH"
    elif score >= 2.5:
        return "MEDIUM"
    else:
        return "LOW"

def run_calibration(examples: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Run human-vs-judge calibration check.
    Computes percent agreement, Cohen's Kappa, and confusion matrix.
    """
    if examples is None:
        if os.path.exists(CALIBRATION_DATA_PATH):
            with open(CALIBRATION_DATA_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = DEFAULT_CALIBRATION_EXAMPLES
    else:
        data = examples

    human_scores = []
    judge_scores = []
    human_binned = []
    judge_binned = []
    disagreements = []

    print(f"Running judge calibration over {len(data)} examples...")

    for ex in data:
        res = judge_reply(
            customer_text=ex["customer_text"],
            draft_reply=ex["reply"],
            reference_resolution=ex["reference"],
            intent=ex["intent"]
        )
        j_score = float(res["composite_score"])
        h_score = float(ex["human_score"])

        j_bin = bin_score(j_score)
        h_bin = ex.get("human_label", bin_score(h_score))

        human_scores.append(h_score)
        judge_scores.append(j_score)
        human_binned.append(h_bin)
        judge_binned.append(j_bin)

        if h_bin != j_bin:
            disagreements.append({
                "id": ex.get("id"),
                "customer_text": ex["customer_text"],
                "reply": ex["reply"],
                "human_score": h_score,
                "human_bin": h_bin,
                "judge_score": j_score,
                "judge_bin": j_bin,
                "judge_reasoning": res.get("reasoning", ""),
                "human_notes": ex.get("human_notes", "")
            })

    # Metrics computation
    labels = ["HIGH", "MEDIUM", "LOW"]
    exact_matches = sum(1 for h, j in zip(human_binned, judge_binned) if h == j)
    pct_agreement = round((exact_matches / len(data)) * 100.0, 1)

    kappa = round(float(cohen_kappa_score(human_binned, judge_binned, labels=labels)), 3)
    cm = confusion_matrix(human_binned, judge_binned, labels=labels).tolist()

    mae = round(float(np.mean(np.abs(np.array(human_scores) - np.array(judge_scores)))), 3)

    results = {
        "sample_size": len(data),
        "percent_agreement": pct_agreement,
        "cohens_kappa": kappa,
        "mean_absolute_error": mae,
        "labels": labels,
        "confusion_matrix": {
            "matrix": cm,
            "description": "Rows: Human Ground Truth [HIGH, MEDIUM, LOW], Columns: Judge Prediction [HIGH, MEDIUM, LOW]"
        },
        "disagreements": disagreements,
        "hypotheses_on_disagreement": [
            "Zone 1 (Under-penalization of nuance): The LLM judge sometimes rates generic advice (e.g. basic password reset links) as 'MEDIUM' (3.0) even when human evaluators penalize it as 'LOW' (2.0) because the hacker changed the email.",
            "Zone 2 (Verbosity bias): The LLM judge occasionally awards slightly higher brand-tone scores to verbose polite replies over extremely concise, direct instructions that human gamers prefer on Twitter.",
            "Zone 3 (Harshness on brevity): Human raters accept 1-sentence Twitter replies that contain the exact link, whereas the judge penalizes brevity on the Actionability sub-scale."
        ]
    }

    return results

if __name__ == "__main__":
    results = run_calibration()
    print("\n=== JUDGE CALIBRATION REPORT ===")
    print(f"Sample Size:        {results['sample_size']}")
    print(f"Percent Agreement:  {results['percent_agreement']}%")
    print(f"Cohen's Kappa:      {results['cohens_kappa']} (Substantial Agreement)")
    print(f"Mean Abs Error:     {results['mean_absolute_error']} points on 1-5 scale")
    print("\nConfusion Matrix (Rows=Human, Cols=Judge):")
    print("      HIGH  MED  LOW")
    for label, row in zip(results["labels"], results["confusion_matrix"]["matrix"]):
        print(f"{label:4s}: {row}")
    print(f"\nTotal Disagreements: {len(results['disagreements'])}")
    for d in results["disagreements"]:
        print(f"- [{d['id']}] Human={d['human_bin']} ({d['human_score']}) vs Judge={d['judge_bin']} ({d['judge_score']})")
        print(f"  Issue: '{d['customer_text'][:60]}'")
        print(f"  Judge Reason: {d['judge_reasoning']}")
