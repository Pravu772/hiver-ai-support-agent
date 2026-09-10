"""
src/responder.py
Generates customer support replies grounded in historical PlayStation support resolutions.
Uses Google Gemini with fallback templates tailored to PlayStation support policies.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intents import get_gemini_client

# Curated fallback responses matching official AskPlayStation handling patterns
FALLBACK_RESPONSES = {
    "account_access_recovery": (
        "Hi there! To regain access to your account or reset your password, please visit "
        "https://www.playstation.com/acct/management. If you suspect your account was compromised, "
        "please reach out directly so our support team can verify your details securely."
    ),
    "account_ban_suspension": (
        "Hello. Account and console suspensions are issued by the PlayStation Safety team following "
        "a thorough review of Code of Conduct violations. These decisions cannot be overturned via Twitter. "
        "Please check your registered email for details or visit playstation.com/safety."
    ),
    "billing_refund_subscription": (
        "Hello! For refund requests on digital PlayStation Store purchases or subscriptions, "
        "please review our cancellation policy and submit a request at https://www.playstation.com/support/store/refunds/. "
        "To manage auto-renew, go to Settings > Account Management > Account Information > PlayStation Subscriptions."
    ),
    "network_outage_connection": (
        "Hi! Please check our real-time service status at https://status.playstation.com. "
        "If all services are green, try power cycling your router and console, or test your connection "
        "under Settings > Network > Test Internet Connection."
    ),
    "hardware_system_crash": (
        "Hello! If your console is freezing or looping, try starting in Safe Mode by holding the power "
        "button until you hear a second beep, then select Option 4 (Rebuild Database). "
        "For hardware repairs, visit https://hardware.support.playstation.com."
    ),
    "game_content_redemption": (
        "Hi there! For missing DLC or voucher issues, please ensure your PSN account region matches the voucher region. "
        "Then go to Settings > Account Management > Restore Licenses to sync your entitlements."
    ),
    "general_inquiry_other": (
        "Hi there! Thanks for reaching out to PlayStation Support. For guides, troubleshooting, "
        "and official announcements, please visit https://www.playstation.com/support."
    )
}

def generate_grounded_reply(
    customer_text: str,
    intent: str,
    retrieved_resolutions: List[Dict[str, Any]],
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a grounded support reply using Gemini.
    Incorporates historical resolved pairs as in-context grounding.
    """
    # Check if Gemini client is available
    client_info = get_gemini_client()

    grounding_context = ""
    if retrieved_resolutions:
        grounding_context = "Historical PlayStation Support Resolutions for Similar Issues:\n"
        for i, res in enumerate(retrieved_resolutions[:3], 1):
            grounding_context += (
                f"[Precedent {i}]\n"
                f"Customer Query: {res.get('customer_query', '')}\n"
                f"Official Reply: {res.get('historical_reply', '')}\n\n"
            )

    if not client_info:
        # Fallback to grounded template / nearest historical reply
        fallback = FALLBACK_RESPONSES.get(intent, FALLBACK_RESPONSES["general_inquiry_other"])
        return {
            "reply": fallback,
            "source": "grounded_template_fallback",
            "grounded_on_count": len(retrieved_resolutions)
        }

    client_type, client = client_info

    prompt = f"""You are PlayStation Support (@AskPlayStation).
Draft ONLY the direct reply tweet to the customer inquiry below.
Guidelines:
- Tone: Friendly, concise, empathetic, and professional (PlayStation brand voice).
- Length: Keep it under 280 characters.
- Grounding: Base your advice strictly on official PlayStation policy. For bans, state policy and tell them to check email/appeals. For refunds or error codes, provide clear guidance.
- Output ONLY the single tweet message. Do NOT include markdown headings, quotes, preamble, or conversational commentary.

Intent: {intent}
{grounding_context}
{"Thread Context: " + context if context else ""}
Customer Message: "{customer_text}"

Reply:"""

    try:
        if client_type == "genai":
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config={"temperature": 0.2}
            )
            reply = response.text.strip()
        else:
            model = client.GenerativeModel("gemini-3.6-flash")
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2}
            )
            reply = response.text.strip()

        # Clean any quotes or prefixes
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1].strip()

        return {
            "reply": reply,
            "source": "gemini",
            "grounded_on_count": len(retrieved_resolutions)
        }
    except Exception as e:
        fallback = FALLBACK_RESPONSES.get(intent, FALLBACK_RESPONSES["general_inquiry_other"])
        return {
            "reply": fallback,
            "source": "fallback_error",
            "error": str(e),
            "grounded_on_count": len(retrieved_resolutions)
        }

if __name__ == "__main__":
    sample_q = "my psn account got banned for no reason"
    res = generate_grounded_reply(sample_q, "account_ban_suspension", [])
    print(f"Customer: {sample_q}")
    print(f"Reply: {res['reply']}")
    print(f"Source: {res['source']}")
