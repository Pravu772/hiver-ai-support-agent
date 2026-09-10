# AskPlayStation AI Customer Support Agent
> **Hiver SDE Intern Assignment**  
> An honest, reproducible, and defensible AI customer support agent for **PlayStation Support (`@AskPlayStation`)** built on real-world multi-turn Twitter customer service data.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](CREDITS.md)
[![Benchmark: 180 Golden Examples](https://img.shields.io/badge/Benchmark-180_Golden_Set-green.svg)](data/sampling_notes.md)

---

## Quickstart: Reproduce Results in Under 2 Minutes

The repository ships with a pre-indexed representative subsample of **3,000 complete AskPlayStation conversation threads** in `data/raw_sample/`. You do **not** need to download the full 492 MB Kaggle dataset to reproduce all headline results.

### 1. Installation
```bash
git clone https://github.com/Pravu772/hiver-ai-support-agent.git
cd hiver-ai-support-agent
pip install -r requirements.txt
```

### 2. (Optional) Set Gemini API Key
To enable live LLM generation with Google Gemini:
```bash
export GEMINI_API_KEY="your-api-key-here"     # Linux / macOS
set GEMINI_API_KEY="your-api-key-here"        # Windows Command Prompt
$env:GEMINI_API_KEY="your-api-key-here"      # Windows PowerShell
```
*(Note: If `GEMINI_API_KEY` is not set, the pipeline automatically uses the built-in deterministic heuristic fallback, ensuring 100% offline reproducibility).*

### 3. Run a Live Test Query
```bash
python cli.py run --text "my psn account got banned for no reason"
```
**Output:**
```
Intent:         account_ban_suspension
Grounded reply: @AskPlayStation Sorry to hear about this. Please check the email address associated
                with your account for details regarding the suspension. You can also find more
                information here: https://www.playstation.com/support/account/psn-suspension-info/
Decision:       ESCALATE
Reason:         Account and console suspensions involve Trust & Safety policy enforcement and
                cannot be automated; requires human safety team review.
```

### 4. Reproduce the Full 3-Way Benchmark
Run the full 180-example evaluation harness comparing the Full Pipeline against the Simple Baseline and Trivial Baseline:
```bash
python cli.py eval
```
*(With a Gemini API key this runs in ~21 minutes and outputs the complete comparison table below. The heuristic fallback completes in under 1 second.)*

---

## Benchmark Headline Results (180 Golden Examples)

| Metric Dimension | Full Pipeline | Simple Baseline (Kwd + Retr) | Trivial Baseline (Majority) |
|---|:---:|:---:|:---:|
| **Intent Classification Accuracy** | 69.4% | **93.3%** | 10.6% |
| **Composite Quality Score (1–5)** | **3.88** | 3.41 | 4.00 |
| — *Groundedness (1–5)* | **4.06** | 3.12 | 4.50 |
| — *Technical Correctness (1–5)* | **3.45** | 3.19 | 3.00 |
| — *PlayStation Tone (1–5)* | **4.51** | 3.44 | 4.00 |
| — *Actionability (1–5)* | 3.50 | **3.90** | 4.50 |
| **Escalation Precision** | **100.0%** | **100.0%** | 0.0% |
| **Escalation Recall** | **100.0%** | 86.8% | 0.0% |
| **False Auto-Handles (Unsafe Leaks)** | **0** | 5 | 38 |
| **False Escalations (Alarms)** | **0** | **0** | **0** |
| **Total Asymmetric Loss ($5\times\text{FN} + 1\times\text{FP}$)** | **0.0** | 25.0 | 190.0 |
| **Loss per Query** | **0.000** | 0.139 | 1.056 |
| **Evaluation Time** | ~21 min (LLM) | ~21 min (LLM) | < 1s |

> [!IMPORTANT]
> **What is misleading about our headline metric?** Read our mandatory critical self-interrogation in [report/REPORT.md](report/REPORT.md), explaining why perfect routing (100%/100%) coexists with only 69.4% intent accuracy, the structural model-fallback degradation, and judge calibration bias.

---

## CLI Usage Guide

The unified `cli.py` interface supports subcommands for isolated component testing and full evaluation:

```bash
# 1. Full end-to-end pipeline execution (Live LLM via Gemini)
python cli.py run --text "my psn account got banned for no reason"

# 2. Instant sub-millisecond execution (offline deterministic fallback)
python cli.py run --fast --text "my psn account got banned for no reason"

# 3. Add -v / --verbose to view top-k historical retrieved precedents and confidence
python cli.py run --text "playstation network error ws-37397-9" -v

# 4. Isolated intent classification
python cli.py classify --text "cannot sign in forgot password"
# (or instant: python cli.py classify --fast --text "cannot sign in forgot password")

# 5. Isolated escalation router test (runs in < 1ms)
python cli.py route --text "i want a refund for the game i bought"

# 6. Isolated grounded reply drafting
python cli.py respond --text "how to restore licenses"
# (or instant: python cli.py respond --fast --text "how to restore licenses")

# 7. Run full evaluation harness
python cli.py eval

# 8. Run judge calibration check
python eval/judge_calibration.py
```

---

## Architecture Overview

```
Customer Message
       │
       ▼
┌─────────────────┐
│ src/intents.py  │ ──► Classifies into 7 derived operational intents (Gemini / Heuristics)
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ src/retrieval.py │ ──► Retrieves top-k similar resolved threads via fast inverted TF-IDF index
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ src/responder.py │ ──► Drafts concise, grounded, brand-compliant reply (Gemini / Fallback)
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│  src/router.py  │ ──► Evaluates safety policy rules + confidence threshold -> AUTO_HANDLE / ESCALATE
└────────┬────────┘
         │
         ▼
Final Output (Intent, Grounded Reply, Decision, Stated Reason)
```

---

## Intent Taxonomy (Derived from Real Data)

Derived empirically from 18,407 reconstructed AskPlayStation conversation threads:
1. `account_access_recovery`: Password resets, 2FA, compromised accounts.
2. `account_ban_suspension`: Policy enforcement, code of conduct violations, ban appeals.
3. `billing_refund_subscription`: PS Store purchases, wallet top-ups, refund requests, auto-renew toggles.
4. `network_outage_connection`: PSN status (`status.playstation.com`), DNS/MTU, error codes (`WS-37397-9`).
5. `hardware_system_crash`: Console freezing, Safe Mode loops, database rebuilds, disc eject failures.
6. `game_content_redemption`: 12-digit vouchers, missing DLC, license restoration (`Restore Licenses`).
7. `general_inquiry_other`: General feedback, backwards compatibility questions, non-support chatter.

---

## Repository Structure

```
hiver-support-agent/
├── data/
│   ├── raw_sample/              # 3,000 reconstructed threads (CSV + JSONL)
│   ├── golden_eval.jsonl        # 180 hand-labelled evaluation examples
│   └── sampling_notes.md        # Stratification, de-duplication, and sampling notes
├── src/
│   ├── ingest.py                # Thread reconstruction & fast data loader
│   ├── intents.py               # Intent taxonomy & classifier (Gemini + fallback)
│   ├── retrieval.py             # Ultra-fast inverted-index TF-IDF retriever (<2ms)
│   ├── responder.py             # Grounded reply generator (PlayStation brand voice)
│   ├── router.py                # Hybrid escalation router with inspectable reasons
│   └── pipeline.py              # Orchestrator connecting all stages
├── eval/
│   ├── baselines.py             # Trivial (Majority) & Simple (Keyword+Retrieval) baselines
│   ├── judge.py                 # LLM-as-a-judge rubric (Groundedness, Correctness, Tone, Actionability)
│   ├── judge_calibration.py     # Human-vs-Judge calibration (35 examples, Cohen's Kappa)
│   ├── calibration_set.json     # Calibrated human ground-truth records
│   ├── run_eval.py              # 3-way evaluation harness comparing all systems
│   └── eval_results.json        # Benchmark output & real failure records
├── report/
│   ├── REPORT.md                # 6-page technical report with failure analysis & limitations
│   └── DECISION_LOG.md          # 12 engineering decision rationales
├── cli.py                       # Unified command-line interface
├── requirements.txt             # Python dependencies
├── CREDITS.md                   # Formal citations for dataset and algorithms
└── README.md                    # Project documentation & reproduction guide
```

---

## Dataset Notice & Provenance
The raw dataset `twcs.csv` (~2.81 million tweets, 492 MB) originates from the public Kaggle repository [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
- Per the assignment instructions, the 492 MB CSV is **excluded** from Git via `.gitignore`.
- This repository ships only `data/raw_sample/askplaystation_threads.csv` (3,000 threads, 1.74 MB), providing 100% reproducibility in seconds without multi-gigabyte downloads.

---

## Credits & License
Created as part of the Hiver SDE Intern Selection Round. Licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](CREDITS.md). External libraries and datasets cited in [CREDITS.md](CREDITS.md).
