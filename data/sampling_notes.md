# Golden Evaluation Set Sampling Notes

## 1. Dataset Origin & Brand Selection
- **Primary Source**: `twcs.csv` from Kaggle's [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (~2.81 million tweets).
- **Brand Chosen**: `AskPlayStation` (PlayStation customer support official Twitter handle).
- **Raw Brand Volume**: 19,098 total tweets authored by `AskPlayStation`, with 22,373 customer tweets mentioning `@AskPlayStation`.
- **Thread Completeness**: 18,407 complete conversation threads were successfully reconstructed by walking backward through `in_response_to_tweet_id` pointers. Broken or orphaned threads (19 total) were discarded.

---

## 2. Representative Subsample Selection
To ensure reproducible execution within 15 minutes and maintain repository cleanliness:
- A representative subsample of **3,000 reconstructed threads** was sampled using a fixed pseudorandom seed (`random_state=42`).
- Stored as `data/raw_sample/askplaystation_threads.csv` (1.74 MB) and `data/raw_sample/askplaystation_threads.jsonl` (2.18 MB).
- The full 492 MB `twcs.csv` is excluded via `.gitignore` and never committed.

---

## 3. Golden Set Stratification Methodology
The golden evaluation set contains **180 real customer examples** sampled from unique conversation threads across the 7 derived intent categories.

### Intent Target Allocation
| Intent Name | Sampled Count | Natural Corpus Frequency | Stratification Rationale |
|---|---|---|---|
| `account_access_recovery` | 30 | 9.3% | Core support surface; critical self-service vs compromise triage. |
| `account_ban_suspension` | 25 | 2.8% | **Deliberately oversampled** (9x relative frequency) to stress-test escalation safety. Unsafe automated handling of ban appeals poses brand risk. |
| `billing_refund_subscription` | 30 | 12.3% | High financial risk; separates auto-handlable subscription toggles from escalation-worthy refund requests. |
| `network_outage_connection` | 30 | 11.5% | Very common on Twitter; tests retrieval grounding against error codes (WS-, CE-, NP-). |
| `hardware_system_crash` | 25 | 3.2% | **Deliberately oversampled** to evaluate Safe Mode guidance vs hardware replacement escalation. |
| `game_content_redemption` | 20 | 3.4% | Tests license restoration and regional voucher handling. |
| `general_inquiry_other` | 20 | 56.1% | **Deliberately undersampled** so the benchmark is not inflated by trivial chit-chat or generic deflections. |
| **Total** | **180** | **100.0%** | Comprehensive, balanced test suite. |

---

## 4. De-duplication & Thread Context
1. **Near-Duplicate Filtering**: Customer query texts were normalized (lowercase, punctuation stripped, whitespace collapsed) to discard near-identical bot inquiries or viral retweet spam.
2. **Multi-Turn Context Preservation**: For multi-turn conversations (thread depth > 2), the golden set preserves the `full_thread_context` alongside the immediate customer message, ensuring context-dependent references (e.g. "I already tried that and it still doesn't work") are resolvable.
3. **Historical Grounding**: Every example includes the actual `historical_brand_reply` sent by `AskPlayStation`, providing an empirical anchor for reference resolutions.

---

## 5. Labelling Schema
Each record in `golden_eval.jsonl` contains:
- `example_id`: Unique identifier (`eval_001` through `eval_180`).
- `customer_tweet_id`: Original tweet ID for traceability.
- `customer_text`: The customer's message text.
- `thread_depth`: Number of turns in the conversation.
- `full_thread_context`: Full chronological multi-turn transcript.
- `historical_brand_reply`: PlayStation's actual tweet reply.
- `intent`: One of the 7 taxonomy intents (hand-verified by candidate).
- `reference_resolution`: Authoritative handling direction based on PlayStation support protocol.
- `decision`: Ground-truth routing decision (`AUTO_HANDLE` vs `ESCALATE`).
- `decision_reasoning`: Inspectable justification for the routing choice.
