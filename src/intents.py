"""
src/intents.py
Intent taxonomy definitions and intent classifier for AskPlayStation support queries.
Leverages Google Gemini API with few-shot prompting, and includes a fallback heuristic classifier.
"""

import os
import json
import re
from typing import Dict, Any, Optional
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Formal Intent Taxonomy derived empirically from AskPlayStation dataset
INTENT_TAXONOMY = {
    "account_access_recovery": {
        "description": "Password resets, sign-in failures, 2-step verification (2FA), compromised or locked accounts.",
        "keywords": ["password", "sign in", "login", "2fa", "two-step", "verification code", "hacked", "stolen", "change email", "reset", "cant access"]
    },
    "account_ban_suspension": {
        "description": "Account or console bans, code of conduct suspensions, ban appeal requests, error code WS-37368-7.",
        "keywords": ["ban", "banned", "suspension", "suspended", "code of conduct", "unban", "terminated", "ws-37368-7"]
    },
    "billing_refund_subscription": {
        "description": "Charges, purchase receipts, refunds for digital games/DLC, PlayStation Plus / PS Now renewals or wallet funds.",
        "keywords": ["refund", "charged", "charge", "purchase", "bought", "subscription", "ps plus", "wallet", "money", "renew", "receipt", "payment"]
    },
    "network_outage_connection": {
        "description": "PSN server status, connectivity failures, DNS/MTU errors, matchmaking dropouts, error codes (WS-37397-9, CE-33987-0, NW-31201-7).",
        "keywords": ["ws-", "ce-33", "np-", "nw-", "error code", "down", "server", "maintenance", "cannot connect", "cant connect", "dns", "mtu", "nat type", "connection", "offline", "lag"]
    },
    "hardware_system_crash": {
        "description": "Console freezing, Safe Mode loops, disc drive ejection failures, overheating, blinking blue/orange lights, rebuilding database.",
        "keywords": ["safe mode", "rebuild database", "freeze", "freezing", "crashed", "crash", "disc", "turning off", "beep", "blue light", "overheating", "hdmi", "fan", "headset", "controller"]
    },
    "game_content_redemption": {
        "description": "Redeeming voucher/promo codes, missing pre-order bonuses/DLC, restore licenses in settings, download queue stuck.",
        "keywords": ["voucher", "code", "redeem", "dlc", "pre-order", "pre order", "restore licenses", "download queue", "addon", "season pass"]
    },
    "general_inquiry_other": {
        "description": "Game release dates, backward compatibility questions, general community feedback, casual greetings or out-of-scope inquiries.",
        "keywords": ["when", "release", "update", "feedback", "hello", "hi", "thank", "gameplay"]
    }
}

VALID_INTENTS = list(INTENT_TAXONOMY.keys())

def classify_intent_heuristic(text: str) -> Dict[str, Any]:
    """
    Deterministic rule-based keyword classifier for baseline / fallback.
    """
    t = text.lower()
    
    # Priority order for safety: bans > access > billing > network > hardware > redemption > general
    for kw in INTENT_TAXONOMY["account_ban_suspension"]["keywords"]:
        if kw in t:
            return {"intent": "account_ban_suspension", "confidence": 0.90, "source": "heuristic_rule"}

    for kw in INTENT_TAXONOMY["account_access_recovery"]["keywords"]:
        if kw in t:
            return {"intent": "account_access_recovery", "confidence": 0.85, "source": "heuristic_rule"}

    for kw in INTENT_TAXONOMY["billing_refund_subscription"]["keywords"]:
        if kw in t:
            return {"intent": "billing_refund_subscription", "confidence": 0.85, "source": "heuristic_rule"}

    for kw in INTENT_TAXONOMY["network_outage_connection"]["keywords"]:
        if kw in t:
            return {"intent": "network_outage_connection", "confidence": 0.85, "source": "heuristic_rule"}

    for kw in INTENT_TAXONOMY["hardware_system_crash"]["keywords"]:
        if kw in t:
            return {"intent": "hardware_system_crash", "confidence": 0.80, "source": "heuristic_rule"}

    for kw in INTENT_TAXONOMY["game_content_redemption"]["keywords"]:
        if kw in t:
            return {"intent": "game_content_redemption", "confidence": 0.80, "source": "heuristic_rule"}

    return {"intent": "general_inquiry_other", "confidence": 0.60, "source": "heuristic_rule"}

def get_gemini_client():
    """
    Initialize Gemini client using either google-genai or google.generativeai.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return ("genai", client)
    except ImportError:
        pass

    try:
        import google.generativeai as gai
        gai.configure(api_key=api_key)
        return ("google.generativeai", gai)
    except ImportError:
        pass

    return None

def classify_intent_gemini(text: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify customer message using Google Gemini LLM with structured output.
    Falls back to heuristic if API key is not present or error occurs.
    """
    client_info = get_gemini_client()
    if not client_info:
        # Fallback to heuristic
        res = classify_intent_heuristic(text)
        res["note"] = "GEMINI_API_KEY not set or client unavailable; used heuristic fallback"
        return res

    client_type, client = client_info

    prompt = f"""You are an expert customer support classifier for PlayStation Support (@AskPlayStation).
Categorize the incoming customer message into exactly ONE of the following valid intents:
- account_access_recovery: Password resets, 2FA, hacked/stolen accounts, sign-in failures.
- account_ban_suspension: Account bans, console suspensions, Code of Conduct violations, ban appeals.
- billing_refund_subscription: Unauthorized charges, refund requests, PS Plus / PS Now renewals, wallet funds.
- network_outage_connection: PSN server down, connection errors (WS-, CE-), DNS/MTU issues.
- hardware_system_crash: Safe Mode loops, console freezing, disc drive issues, crash/rebuild database.
- game_content_redemption: Voucher codes, missing DLC/pre-order items, restore licenses.
- general_inquiry_other: General questions, game release info, feedback, greetings.

Customer Message:
"{text}"

{"Thread Context:" if context else ""}
{context or ""}

Respond ONLY with valid JSON in this exact structure:
{{
  "intent": "<one of the 7 valid intent names>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<1-sentence explanation>"
}}
"""

    try:
        if client_type == "genai":
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config={"temperature": 0.0, "response_mime_type": "application/json"}
            )
            raw = response.text
        else:
            model = client.GenerativeModel("gemini-3.6-flash")
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0}
            )
            raw = response.text

        # Parse JSON
        clean_json = re.search(r"\{.*\}", raw, re.DOTALL)
        if clean_json:
            data = json.loads(clean_json.group(0))
            intent = data.get("intent", "").strip()
            if intent in VALID_INTENTS:
                return {
                    "intent": intent,
                    "confidence": float(data.get("confidence", 0.9)),
                    "reasoning": data.get("reasoning", ""),
                    "source": "gemini"
                }
    except Exception as e:
        fallback = classify_intent_heuristic(text)
        fallback["error"] = str(e)
        return fallback

    # Default fallback if parsing failed
    return classify_intent_heuristic(text)

def classify_intent(text: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Primary entrypoint for intent classification."""
    return classify_intent_gemini(text, context)

if __name__ == "__main__":
    test_queries = [
        "my psn account got banned for no reason please help",
        "cannot sign in forgot my password and my 2fa phone number changed",
        "i was charged twice for playstation plus renewal want a refund",
        "playstation network is down getting error ws-37397-9",
        "my ps4 is stuck in a safe mode loop and turns off",
        "where do i enter my 12 digit voucher code for spiderman"
    ]
    for q in test_queries:
        res = classify_intent(q)
        print(f"Query: {q}")
        print(f"Result: {res['intent']} (conf: {res['confidence']:.2f}, source: {res.get('source')})\n")
