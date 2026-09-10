"""
eval/run_eval.py
Full evaluation harness running over the golden set (180 examples):
Compares:
1. Full Pipeline (Gemini Classifier + TF-IDF Retrieval + Grounded Responder + Safety Router)
2. Simple Baseline (Keyword-rule Classifier + Top-1 Retrieval-only Reply + Keyword Router)
3. Trivial Baseline (Majority-class Classifier + Fixed Canned Response + Always Auto-Handle)

Computes:
- Overall & per-intent classification accuracy
- Grounded reply quality on 4 rubric dimensions (1-5)
- Routing precision, recall, F1, and Asymmetric Routing Cost (penalizing false auto-handles 5x)
"""

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline import get_pipeline
from eval.baselines import TrivialBaseline, SimpleBaseline
from eval.judge import judge_reply, evaluate_reply_heuristic

GOLDEN_EVAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden_eval.jsonl")
RESULTS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "eval_results.json")

def load_golden_set() -> List[Dict[str, Any]]:
    if not os.path.exists(GOLDEN_EVAL_PATH):
        raise FileNotFoundError(f"Golden eval set not found at {GOLDEN_EVAL_PATH}")
    records = []
    with open(GOLDEN_EVAL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

def calculate_routing_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Compute Precision, Recall, and F1 for ESCALATE and AUTO_HANDLE,
    plus asymmetric cost where False Auto-Handle (missed escalation) = 5.0, False Escalate = 1.0.
    """
    tp_esc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "ESCALATE" and yp == "ESCALATE")
    fp_esc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "AUTO_HANDLE" and yp == "ESCALATE")
    fn_esc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "ESCALATE" and yp == "AUTO_HANDLE")
    tn_esc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "AUTO_HANDLE" and yp == "AUTO_HANDLE")

    prec_esc = tp_esc / (tp_esc + fp_esc) if (tp_esc + fp_esc) > 0 else 0.0
    rec_esc = tp_esc / (tp_esc + fn_esc) if (tp_esc + fn_esc) > 0 else 0.0
    f1_esc = 2 * prec_esc * rec_esc / (prec_esc + rec_esc) if (prec_esc + rec_esc) > 0 else 0.0

    prec_auto = tn_esc / (tn_esc + fn_esc) if (tn_esc + fn_esc) > 0 else 0.0
    rec_auto = tn_esc / (tn_esc + fp_esc) if (tn_esc + fp_esc) > 0 else 0.0
    f1_auto = 2 * prec_auto * rec_auto / (prec_auto + rec_auto) if (prec_auto + rec_auto) > 0 else 0.0

    # Asymmetric Cost: 5x penalty for missed escalation (fn_esc), 1x penalty for false alarm (fp_esc)
    asymmetric_cost = (fn_esc * 5.0) + (fp_esc * 1.0)
    normalized_cost = asymmetric_cost / len(y_true)

    return {
        "escalate_precision": round(prec_esc * 100, 1),
        "escalate_recall": round(rec_esc * 100, 1),
        "escalate_f1": round(f1_esc * 100, 1),
        "auto_precision": round(prec_auto * 100, 1),
        "auto_recall": round(rec_auto * 100, 1),
        "auto_f1": round(f1_auto * 100, 1),
        "false_auto_handles": fn_esc,
        "false_escalations": fp_esc,
        "asymmetric_cost_total": asymmetric_cost,
        "asymmetric_cost_per_query": round(normalized_cost, 3)
    }

def run_evaluation(limit: Optional[int] = None) -> Dict[str, Any]:
    golden_data = load_golden_set()
    if limit:
        golden_data = golden_data[:limit]

    print(f"Loaded {len(golden_data)} golden evaluation examples.")
    print("Initializing Pipeline, Simple Baseline, and Trivial Baseline...")

    pipeline = get_pipeline()
    simple_base = SimpleBaseline()
    trivial_base = TrivialBaseline()

    systems = {
        "Full Pipeline": {"predictions": [], "intents": [], "decisions": [], "judge_scores": []},
        "Simple Baseline": {"predictions": [], "intents": [], "decisions": [], "judge_scores": []},
        "Trivial Baseline": {"predictions": [], "intents": [], "decisions": [], "judge_scores": []}
    }

    y_true_intents = [ex["intent"] for ex in golden_data]
    y_true_decisions = [ex["decision"] for ex in golden_data]

    print("Running evaluation across all 3 systems...")
    t0 = time.time()

    failures = []
    sample_evaluations = []

    from concurrent.futures import ThreadPoolExecutor

    def eval_item(ex):
        q = ex["customer_text"]
        ref_res = ex["reference_resolution"]
        true_intent = ex["intent"]
        true_dec = ex["decision"]
        context = ex.get("full_thread_context", "")

        pipe_res = pipeline.run(q, context=context)
        pipe_judge = judge_reply(q, pipe_res["grounded_reply"], ref_res, pipe_res["intent"])

        simp_res = simple_base.run(q, context=context)
        simp_judge = evaluate_reply_heuristic(q, simp_res["grounded_reply"], ref_res, simp_res["intent"])

        triv_res = trivial_base.run(q, context=context)
        triv_judge = evaluate_reply_heuristic(q, triv_res["grounded_reply"], ref_res, triv_res["intent"])

        failure = None
        if pipe_res["intent"] != true_intent or pipe_res["decision"] != true_dec:
            failure = {
                "example_id": ex["example_id"],
                "customer_text": q,
                "true_intent": true_intent,
                "pred_intent": pipe_res["intent"],
                "true_decision": true_dec,
                "pred_decision": pipe_res["decision"],
                "routing_reason": pipe_res["reason"],
                "draft_reply": pipe_res["grounded_reply"],
                "historical_reply": ex["historical_brand_reply"]
            }

        sample_eval = {
            "example_id": ex["example_id"],
            "customer_text": q,
            "true_intent": true_intent,
            "pred_intent": pipe_res["intent"],
            "pred_decision": pipe_res["decision"],
            "grounded_reply": pipe_res["grounded_reply"],
            "judge_source": pipe_judge.get("judge_source"),
            "judge_scores": {
                "groundedness": pipe_judge.get("groundedness"),
                "correctness": pipe_judge.get("correctness"),
                "tone": pipe_judge.get("tone"),
                "actionability": pipe_judge.get("actionability"),
                "composite": pipe_judge.get("composite_score")
            },
            "judge_reasoning": pipe_judge.get("reasoning")
        }

        return {
            "example_id": ex["example_id"],
            "pipe_res": pipe_res,
            "pipe_judge": pipe_judge,
            "simp_res": simp_res,
            "simp_judge": simp_judge,
            "triv_res": triv_res,
            "triv_judge": triv_judge,
            "failure": failure,
            "sample_eval": sample_eval
        }

    completed_count = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        for item in executor.map(eval_item, golden_data):
            completed_count += 1
            systems["Full Pipeline"]["intents"].append(item["pipe_res"]["intent"])
            systems["Full Pipeline"]["decisions"].append(item["pipe_res"]["decision"])
            systems["Full Pipeline"]["predictions"].append(item["pipe_res"]["grounded_reply"])
            systems["Full Pipeline"]["judge_scores"].append(item["pipe_judge"])

            systems["Simple Baseline"]["intents"].append(item["simp_res"]["intent"])
            systems["Simple Baseline"]["decisions"].append(item["simp_res"]["decision"])
            systems["Simple Baseline"]["predictions"].append(item["simp_res"]["grounded_reply"])
            systems["Simple Baseline"]["judge_scores"].append(item["simp_judge"])

            systems["Trivial Baseline"]["intents"].append(item["triv_res"]["intent"])
            systems["Trivial Baseline"]["decisions"].append(item["triv_res"]["decision"])
            systems["Trivial Baseline"]["predictions"].append(item["triv_res"]["grounded_reply"])
            systems["Trivial Baseline"]["judge_scores"].append(item["triv_judge"])

            if item["failure"]:
                failures.append(item["failure"])
            if len(sample_evaluations) < 15:
                sample_evaluations.append(item["sample_eval"])

            if completed_count % 10 == 0 or completed_count == len(golden_data):
                print(f"Evaluated {completed_count}/{len(golden_data)} examples ({time.time() - t0:.1f}s)...", flush=True)

    # Compute Summary Statistics
    summary = {}
    for name, data in systems.items():
        # Accuracy
        acc = sum(1 for yt, yp in zip(y_true_intents, data["intents"]) if yt == yp) / len(y_true_intents)
        
        # Per-class breakdown
        class_breakdown = {}
        unique_intents = sorted(list(set(y_true_intents)))
        for intent in unique_intents:
            idx_list = [idx for idx, yt in enumerate(y_true_intents) if yt == intent]
            class_correct = sum(1 for idx in idx_list if data["intents"][idx] == intent)
            class_acc = round((class_correct / len(idx_list)) * 100, 1) if idx_list else 0.0
            class_breakdown[intent] = {
                "count": len(idx_list),
                "accuracy": class_acc
            }

        # Quality Scores
        g_scores = [s["groundedness"] for s in data["judge_scores"]]
        c_scores = [s["correctness"] for s in data["judge_scores"]]
        t_scores = [s["tone"] for s in data["judge_scores"]]
        a_scores = [s["actionability"] for s in data["judge_scores"]]
        comp_scores = [s["composite_score"] for s in data["judge_scores"]]

        # Routing Metrics
        r_metrics = calculate_routing_metrics(y_true_decisions, data["decisions"])

        summary[name] = {
            "intent_accuracy": round(acc * 100, 1),
            "class_breakdown": class_breakdown,
            "quality_groundedness": round(float(np.mean(g_scores)), 2),
            "quality_correctness": round(float(np.mean(c_scores)), 2),
            "quality_tone": round(float(np.mean(t_scores)), 2),
            "quality_actionability": round(float(np.mean(a_scores)), 2),
            "quality_composite": round(float(np.mean(comp_scores)), 2),
            "routing": r_metrics
        }

    eval_output = {
        "metadata": {
            "eval_size": len(golden_data),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.time() - t0, 2)
        },
        "comparison_table": summary,
        "sample_evaluations": sample_evaluations,
        "sample_failures": failures[:15]
    }

    # Save to disk
    with open(RESULTS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation complete in {eval_output['metadata']['elapsed_seconds']}s! Results saved to {RESULTS_OUTPUT_PATH}")
    return eval_output

def print_results_table(results: Dict[str, Any]):
    """Print formatted markdown comparison table."""
    comp = results["comparison_table"]
    
    print("\n" + "="*80)
    print("   THREE-WAY BENCHMARK EVALUATION RESULTS (180 GOLDEN EXAMPLES)")
    print("="*80)
    
    headers = [
        "Metric Dimension",
        "Full Pipeline",
        "Simple Baseline (Kwd+Retr)",
        "Trivial Baseline (Majority)"
    ]
    
    rows = [
        ("Intent Accuracy", f"{comp['Full Pipeline']['intent_accuracy']}%", f"{comp['Simple Baseline']['intent_accuracy']}%", f"{comp['Trivial Baseline']['intent_accuracy']}%"),
        ("Composite Quality (1-5)", f"{comp['Full Pipeline']['quality_composite']}", f"{comp['Simple Baseline']['quality_composite']}", f"{comp['Trivial Baseline']['quality_composite']}"),
        (" - Groundedness (1-5)", f"{comp['Full Pipeline']['quality_groundedness']}", f"{comp['Simple Baseline']['quality_groundedness']}", f"{comp['Trivial Baseline']['quality_groundedness']}"),
        (" - Correctness (1-5)", f"{comp['Full Pipeline']['quality_correctness']}", f"{comp['Simple Baseline']['quality_correctness']}", f"{comp['Trivial Baseline']['quality_correctness']}"),
        (" - PlayStation Tone (1-5)", f"{comp['Full Pipeline']['quality_tone']}", f"{comp['Simple Baseline']['quality_tone']}", f"{comp['Trivial Baseline']['quality_tone']}"),
        (" - Actionability (1-5)", f"{comp['Full Pipeline']['quality_actionability']}", f"{comp['Simple Baseline']['quality_actionability']}", f"{comp['Trivial Baseline']['quality_actionability']}"),
        ("Escalate Precision", f"{comp['Full Pipeline']['routing']['escalate_precision']}%", f"{comp['Simple Baseline']['routing']['escalate_precision']}%", f"{comp['Trivial Baseline']['routing']['escalate_precision']}%"),
        ("Escalate Recall", f"{comp['Full Pipeline']['routing']['escalate_recall']}%", f"{comp['Simple Baseline']['routing']['escalate_recall']}%", f"{comp['Trivial Baseline']['routing']['escalate_recall']}%"),
        ("False Auto-Handles (Missed)", f"{comp['Full Pipeline']['routing']['false_auto_handles']}", f"{comp['Simple Baseline']['routing']['false_auto_handles']}", f"{comp['Trivial Baseline']['routing']['false_auto_handles']}"),
        ("False Escalations (Alarms)", f"{comp['Full Pipeline']['routing']['false_escalations']}", f"{comp['Simple Baseline']['routing']['false_escalations']}", f"{comp['Trivial Baseline']['routing']['false_escalations']}"),
        ("Asymmetric Loss (5x Miss)", f"{comp['Full Pipeline']['routing']['asymmetric_cost_total']}", f"{comp['Simple Baseline']['routing']['asymmetric_cost_total']}", f"{comp['Trivial Baseline']['routing']['asymmetric_cost_total']}"),
        ("Loss / Query", f"{comp['Full Pipeline']['routing']['asymmetric_cost_per_query']}", f"{comp['Simple Baseline']['routing']['asymmetric_cost_per_query']}", f"{comp['Trivial Baseline']['routing']['asymmetric_cost_per_query']}")
    ]

    fmt = "{:<28} | {:<16} | {:<27} | {:<24}"
    print(fmt.format(*headers))
    print("-" * 105)
    for r in rows:
        print(fmt.format(*r))
    print("="*105)

    print("\n--- Per-Class Intent Accuracy (Full Pipeline) ---")
    cb = comp['Full Pipeline']['class_breakdown']
    for intent, stats in cb.items():
        print(f"{intent:<30}: {stats['accuracy']:>5.1f}% (n={stats['count']})")
    print()

if __name__ == "__main__":
    results = run_evaluation()
    print_results_table(results)
