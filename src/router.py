"""
src/router.py
Escalation router deciding whether a customer message can be AUTO_HANDLE or must ESCALATE.
Combines rule-informed safety policies with confidence thresholds and produces inspectable textual reasons.
"""

import re
from typing import Dict, Any, Optional

def route_message(
    customer_text: str,
    intent: str,
    confidence: float = 0.85,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Decide whether to AUTO_HANDLE or ESCALATE the customer inquiry.
    Returns:
        {
            "decision": "AUTO_HANDLE" | "ESCALATE",
            "reason": "<clear textual explanation of why>",
            "policy_rule_triggered": Optional[str]
        }
    """
    text_lower = customer_text.lower()

    # Rule 1: Account Bans and Suspensions (Strict Brand Safety Policy)
    if intent == "account_ban_suspension" or any(k in text_lower for k in ["ban", "banned", "suspension", "suspended", "code of conduct"]):
        return {
            "decision": "ESCALATE",
            "reason": "Account and console suspensions involve Trust & Safety policy enforcement and cannot be automated; requires human safety team review.",
            "policy_rule_triggered": "SAFETY_BAN_POLICY"
        }

    # Rule 2: Account Compromise / Hacking / Security Breach
    if any(k in text_lower for k in ["hacked", "stolen", "someone changed", "unauthorized access", "someone compromised", "someone logged into"]):
        return {
            "decision": "ESCALATE",
            "reason": "Suspected account compromise requires sensitive identity verification by a human support specialist.",
            "policy_rule_triggered": "SECURITY_COMPROMISE_POLICY"
        }

    # Rule 3: Financial Disputes & Direct Refund Requests
    if intent == "billing_refund_subscription":
        if any(k in text_lower for k in ["refund", "double charged", "charged twice", "money back", "unauthorized charge", "cancel and refund", "bank dispute", "fraud"]):
            return {
                "decision": "ESCALATE",
                "reason": "Financial refunds and billing disputes require payment verification and human agent authorization.",
                "policy_rule_triggered": "FINANCIAL_REFUND_POLICY"
            }

    # Rule 4: Physical Hardware Repair / Severe Damage
    if intent == "hardware_system_crash":
        if any(k in text_lower for k in ["smoke", "spark", "smell", "burned", "melted", "warranty repair", "send it in", "broken disc drive", "hardware replacement"]):
            return {
                "decision": "ESCALATE",
                "reason": "Physical console hardware failures and warranty repair claims require RMA escalation.",
                "policy_rule_triggered": "HARDWARE_REPAIR_POLICY"
            }

    # Rule 5: Low Confidence / High Ambiguity
    if confidence < 0.65:
        return {
            "decision": "ESCALATE",
            "reason": f"Intent classification confidence ({confidence:.2f}) is below the automated safety threshold (0.65); routed to human agent.",
            "policy_rule_triggered": "LOW_CONFIDENCE_THRESHOLD"
        }

    # Default Rule 6: Standard Self-Service Procedures
    handling_reasons = {
        "account_access_recovery": "Standard self-service password recovery flow via official web portal.",
        "billing_refund_subscription": "Self-service subscription management and auto-renewal toggle instructions.",
        "network_outage_connection": "Automated network status check (status.playstation.com) and standard router/DNS troubleshooting.",
        "hardware_system_crash": "Standard self-service Safe Mode database rebuild and troubleshooting guidance.",
        "game_content_redemption": "Standard self-service license restoration (Settings > Account Management > Restore Licenses).",
        "general_inquiry_other": "Public product FAQ and general information inquiry."
    }

    reason = handling_reasons.get(
        intent,
        "Issue matches standard documented troubleshooting procedures and can be safely automated."
    )

    return {
        "decision": "AUTO_HANDLE",
        "reason": reason,
        "policy_rule_triggered": None
    }

if __name__ == "__main__":
    cases = [
        ("my psn account got banned for no reason", "account_ban_suspension", 0.95),
        ("someone hacked my account and changed the 2fa email", "account_access_recovery", 0.90),
        ("i want a refund for the game i bought yesterday", "billing_refund_subscription", 0.90),
        ("how do i turn off auto renew for ps plus", "billing_refund_subscription", 0.85),
        ("is psn down error ws-37397-9", "network_outage_connection", 0.90),
        ("my ps4 is freezing in menu", "hardware_system_crash", 0.85)
    ]
    for text, intent, conf in cases:
        res = route_message(text, intent, conf)
        print(f"Text: {text}")
        print(f"Decision: {res['decision']} | Reason: {res['reason']}\n")
