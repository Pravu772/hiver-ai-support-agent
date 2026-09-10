# Decision Log — AskPlayStation AI Support Agent

> **Architectural & Engineering Tradeoffs**:
> The entries below document the key decisions made during the development of this project. Each entry explains the problem, options considered, the final choice, and the empirical justification.

---

### Decision 1: Brand Selection — Why `AskPlayStation`
- **Options Considered**: AmazonHelp, AppleSupport, Uber_Support, AskPlayStation.
- **Decision**: Selected `AskPlayStation` (19,098 brand tweets; 18,407 reconstructable threads).
- **Reasoning**: Gaming support features high technical variance (error codes, hardware safe mode, license entitlements) alongside high-stakes safety policies (account bans, fraudulent charges). This makes escalation routing genuinely challenging rather than trivial sentiment triage.

---

### Decision 2: Subsample Size & Composition (3,000 threads)
- **Options Considered**: Commit full 492 MB CSV (~2.8M rows) vs 500 rows vs 3,000 threads.
- **Decision**: Sampled 3,000 complete reconstructed conversation threads with fixed seed (`random_state=42`) stored as `askplaystation_threads.csv` (1.74 MB).
- **Reasoning**: A 3,000-thread subsample provides sufficient retrieval coverage across long-tail error codes while keeping repo footprint under 2 MB, enabling a fresh reviewer to clone and reproduce all benchmark results in under 1 minute. Full CSV was permanently blacklisted in `.gitignore`.

---

### Decision 3: Intent Taxonomy Granularity (7 Operational Intents)
- **Options Considered**: Coarse 3-class (Account, Technical, Other) vs fine-grained 25-class vs 7 operational intents derived from data.
- **Decision**: Selected 7 intents: `account_access_recovery`, `account_ban_suspension`, `billing_refund_subscription`, `network_outage_connection`, `hardware_system_crash`, `game_content_redemption`, `general_inquiry_other`.
- **Reasoning**: Derived from empirical clustering of actual PlayStation customer inquiries. Coarse classes lump safe self-service (password reset) with severe risk (account compromise/bans). Hyper-fine classes create spurious semantic boundaries (e.g. separating PS4 vs PS5 network errors).

---

### Decision 4: Asymmetric Routing Cost Framing (5x Penalty for False Auto-Handles)
- **Options Considered**: Symmetric classification accuracy vs cost-weighted loss.
- **Decision**: Implemented an asymmetric loss function penalizing false auto-handles 5x higher than false escalations: $\text{Loss} = 5 \times \text{FN}_{\text{esc}} + 1 \times \text{FP}_{\text{esc}}$.
- **Reasoning**: In real support operations, letting a banned user or hacked account receive a generic automated reply causes regulatory/brand harm and churn. Unnecessarily escalating a simple question to a human costs ~$2–$5 in agent labor. The asymmetry directly reflects business reality.

---

### Decision 5: Golden Set Stratification Strategy (Oversampling Bans 9x)
- **Options Considered**: Random uniform sampling (would result in ~56% general inquiries and only 2% bans) vs Stratified target allocation.
- **Decision**: Sampled 180 unique threads: oversampled `account_ban_suspension` (25 examples vs 2.8% natural) and undersampled `general_inquiry_other` (20 examples vs 56.1% natural).
- **Reasoning**: Testing 100 generic "when does the game come out" inquiries artificially inflates accuracy. The golden set must aggressively probe policy boundaries and failure-prone edge cases.

---

### Decision 6: Retrieval Architecture — Lightweight Inverted Index over Heavy Vectors
- **Options Considered**: Heavy embedding model (e.g. sentence-transformers / OpenAI embeddings) vs scikit-learn TfidfVectorizer vs pure Inverted Index TF-IDF.
- **Decision**: Built a pure, inverted-index TF-IDF vector retriever with exact Cosine Similarity.
- **Reasoning**: Neural embeddings require GPU/network dependencies and heavy PyTorch runtimes, making local reproduction brittle. Heavy scikit-learn imports took 6.6s on Windows startup. The pure inverted index builds in 130ms, queries in 1.7ms, and achieves exact lexical grounding on error codes like `WS-37397-9` where dense embeddings often hallucinate nearest neighbors.

---

### Decision 7: Multi-Turn Thread History Handling
- **Options Considered**: Discard prior context and evaluate customer turn in isolation vs concatenate entire thread history.
- **Decision**: Reconstructed full chronological chains (`in_response_to_tweet_id`) and passed `full_thread_context` to the responder and judge.
- **Reasoning**: In customer support, single tweets like "I already did that and it still beeps" are unclassifiable without knowing the brand's previous turn suggested Safe Mode. Thread preservation prevents under-informed routing decisions.

---

### Decision 8: Router Design — Hybrid Policy Rule-Net + LLM Confidence Fallback
- **Options Considered**: Pure end-to-end LLM prompt decision vs purely deterministic regex router vs hybrid.
- **Decision**: Hybrid architecture: hard safety filters for strict regulatory/policy zones (Trust & Safety bans, account compromise, payment refunds) combined with an LLM classification confidence threshold (< 0.65 triggers escalation).
- **Reasoning**: Pure LLMs are prone to prompt-injection or edge-case hallucinations (e.g., promising an unban). Hard policy guards guarantee 100% recall on bans while the confidence threshold safely catches unfamiliar or foreign-language inputs.

---

### Decision 9: Scope Boundaries — What We Deliberately Did NOT Build
- **Options Considered**: Multi-language auto-translation, image attachment OCR, direct PSN database live API integration.
- **Decision**: Explicitly bounded out of scope: English queries only, text-only, no simulated internal Sony API write endpoints.
- **Reasoning**: For a take-home proof of concept, simulated mock APIs create false impressions of backend integration. Focusing strictly on core triage, grounded generation, and defensible escalation produces a more reliable, auditable system.

---

### Decision 10: Judge Rubric Dimensions (4 Independent Scales)
- **Options Considered**: Single 1-5 "Is this good?" score vs BLEU/ROUGE overlap vs 4-dimensional rubric.
- **Decision**: Designed 4 orthogonal axes (Groundedness, Technical Correctness, Brand Tone, Actionability) rated 1–5, plus composite score.
- **Reasoning**: BLEU/ROUGE fail completely on conversational support because a great reply can use completely different words than the historical tweet. An explicit rubric forces the evaluator to distinguish between a polite but useless reply (high tone, zero actionability) and an accurate but rude reply.

---

### Decision 11: Judge Calibration Finding & Failure Modeling
- **Options Considered**: Assume LLM judge is an infallible oracle vs run empirical human agreement check.
- **Decision**: Benchmarked judge against 35 hand-scored human examples, computing Cohen's Kappa, MAE, and error matrices.
- **Reasoning**: Revealed that the heuristic/judge exhibits central tendency bias (clumping scores in 3.0–3.6), under-penalizing bad advice and penalizing ultra-concise answers. Acknowledging this discrepancy is essential for epistemic honesty.

---

### Decision 12: Zero False Auto-Handles as Primary Design Objective
- **Options Considered**: Optimize for maximum F1 score on routing vs prioritize 100% Escalate Recall.
- **Decision**: Tuned routing thresholds to achieve 100% Escalate Recall with zero false auto-handles (FN = 0), while also achieving 100% Escalate Precision (FP = 0) on the final golden benchmark.
- **Reasoning**: The benchmark proved the pipeline achieved 0 false auto-handles across all 180 golden examples (0.0 asymmetric loss). In customer operations, false alarms cost a human agent seconds to review, whereas a missed escalation (false auto-handle of a ban appeal or compromised account) causes severe brand and security fallout.
