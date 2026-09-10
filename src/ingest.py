"""
src/ingest.py
Handles loading of reconstructed conversation threads and on-the-fly thread
reconstruction from raw Twitter customer support dataset (twcs.csv).
"""

import os
import sys
import re
import pandas as pd
from typing import List, Dict, Optional, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw_sample", "askplaystation_threads.csv"
)
DEFAULT_JSONL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw_sample", "askplaystation_threads.jsonl"
)

def clean_tweet_text(text: str) -> str:
    """Strip leading @mentions, normalize multiple whitespaces."""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"^(@\w+\s*)+", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def load_sample_threads(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fast loader for pre-reconstructed AskPlayStation conversation threads.
    Prioritizes askplaystation_threads.csv for sub-50ms read speed.
    """
    target = file_path or DEFAULT_CSV_PATH
    if not os.path.exists(target):
        target = DEFAULT_JSONL_PATH
    
    if not os.path.exists(target):
        raise FileNotFoundError(
            f"Could not find sample threads at {target}. "
            "Please ensure data/raw_sample/askplaystation_threads.csv exists."
        )

    if target.endswith(".csv"):
        df = pd.read_csv(target)
    else:
        df = pd.read_json(target, lines=True)

    threads = df.to_dict(orient="records")
    return threads

def reconstruct_from_raw_csv(csv_path: str, brand: str = "AskPlayStation") -> List[Dict[str, Any]]:
    """
    Reconstruct full conversation threads from raw twcs.csv file.
    Walks in_response_to_tweet_id backward to link customer messages with brand responses.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Raw CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce")
    df = df.dropna(subset=["tweet_id"])
    df["tweet_id"] = df["tweet_id"].astype(int)

    brand_mask = df["author_id"] == brand
    replies = df[brand_mask & df["in_response_to_tweet_id"].notna()].copy()

    tweet_lookup = df.set_index("tweet_id")[
        ["author_id", "inbound", "created_at", "text", "in_response_to_tweet_id"]
    ].to_dict(orient="index")

    complete_threads = []
    for _, row in replies.iterrows():
        b_id = int(row["tweet_id"])
        try:
            p_id = int(float(row["in_response_to_tweet_id"]))
        except (ValueError, TypeError):
            continue

        chain = []
        curr = p_id
        visited = set()
        while curr and curr in tweet_lookup and curr not in visited:
            visited.add(curr)
            p_row = tweet_lookup[curr]
            chain.append({
                "tweet_id": curr,
                "author_id": str(p_row["author_id"]),
                "inbound": bool(p_row["inbound"]),
                "created_at": str(p_row["created_at"]),
                "text": str(p_row["text"]) if pd.notna(p_row["text"]) else ""
            })
            next_p = p_row["in_response_to_tweet_id"]
            if pd.isna(next_p) or next_p == "":
                break
            try:
                curr = int(float(next_p))
            except (ValueError, TypeError):
                break

        if not chain:
            continue

        chain.reverse()
        last_msg = chain[-1]
        clean_cust_text = clean_tweet_text(last_msg["text"])
        if len(clean_cust_text) < 5:
            continue

        formatted_turns = []
        for turn in chain:
            role = "Customer" if turn["inbound"] else f"Brand ({turn['author_id']})"
            formatted_turns.append(f"{role}: {turn['text']}")
        full_context = "\n".join(formatted_turns)

        complete_threads.append({
            "brand_tweet_id": b_id,
            "brand_reply": str(row["text"]).strip(),
            "customer_tweet_id": last_msg["tweet_id"],
            "customer_text": clean_cust_text,
            "raw_customer_text": last_msg["text"],
            "full_thread_context": full_context,
            "thread_depth": len(chain) + 1,
            "created_at": str(row["created_at"])
        })

    return complete_threads

if __name__ == "__main__":
    threads = load_sample_threads()
    print(f"Loaded {len(threads):,} sample threads.")
    if threads:
        print("Example 1 Customer Text:", threads[0]["customer_text"][:80])
        print("Example 1 Brand Reply:", threads[0]["brand_reply"][:80])
