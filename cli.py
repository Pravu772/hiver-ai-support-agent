#!/usr/bin/env python3
"""
cli.py - Unified CLI entrypoint for the AskPlayStation AI Customer Support Agent.

Subcommands:
  classify  - Classify an incoming customer message into the intent taxonomy
  respond   - Generate a grounded support reply based on historical resolutions
  route     - Decide whether to AUTO_HANDLE or ESCALATE with inspectable reasoning
  run       - Run the full end-to-end support pipeline on a customer query
  eval      - Run the evaluation harness comparing the pipeline against baselines
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def cmd_classify(args):
    """Isolated intent classification."""
    from src.intents import classify_intent
    text = args.text
    context = getattr(args, "context", None)
    result = classify_intent(text, context)
    print(f"Intent:     {result['intent']}")
    print(f"Confidence: {result.get('confidence', 0.85):.2f}")
    if "reasoning" in result and result["reasoning"]:
        print(f"Reasoning:  {result['reasoning']}")
    if "source" in result:
        print(f"Source:     {result['source']}")

def cmd_respond(args):
    """Isolated grounded reply drafting."""
    from src.intents import classify_intent
    from src.retrieval import retrieve_similar_resolutions
    from src.responder import generate_grounded_reply
    text = args.text
    intent = getattr(args, "intent", None)
    if not intent:
        clf = classify_intent(text)
        intent = clf["intent"]
    
    retrieved = retrieve_similar_resolutions(text, top_k=3)
    resp = generate_grounded_reply(text, intent, retrieved)
    print(f"Intent:         {intent}")
    print(f"Grounded reply: {resp['reply']}")
    print(f"Source:         {resp.get('source', 'unknown')}")

def cmd_route(args):
    """Isolated escalation router."""
    from src.intents import classify_intent
    from src.router import route_message
    text = args.text
    intent = getattr(args, "intent", None)
    confidence = getattr(args, "confidence", 0.85)
    if not intent:
        clf = classify_intent(text)
        intent = clf["intent"]
        confidence = clf.get("confidence", 0.85)

    res = route_message(text, intent, confidence)
    print(f"Decision: {res['decision']}")
    print(f"Reason:   {res['reason']}")
    if res.get("policy_rule_triggered"):
        print(f"Policy:   {res['policy_rule_triggered']}")

def cmd_run(args):
    """Full end-to-end pipeline execution."""
    from src.pipeline import run_pipeline
    text = args.text
    context = getattr(args, "context", None)
    res = run_pipeline(text, context)

    print(f"Intent:         {res['intent']}")
    print(f"Grounded reply: {res['grounded_reply']}")
    print(f"Decision:       {res['decision']}")
    print(f"Reason:         {res['reason']}")
    if args.verbose:
        print("\n--- Additional Pipeline Metadata ---")
        print(f"Confidence:     {res['confidence']:.2f}")
        print(f"Reply Source:   {res['reply_source']}")
        if res.get("policy_rule_triggered"):
            print(f"Policy Rule:    {res['policy_rule_triggered']}")
        print(f"\nTop-{len(res['retrieved_precedents'])} Retrieved Historical Precedents:")
        for i, p in enumerate(res['retrieved_precedents'], 1):
            print(f" [{i}] Score: {p['similarity_score']} | Customer: \"{p['customer_query'][:60]}...\"")
            print(f"     Reply: \"{p['historical_reply'][:80]}...\"")

def cmd_eval(args):
    """Run full evaluation harness over the golden set."""
    from eval.run_eval import run_evaluation, print_results_table
    limit = getattr(args, "limit", None)
    results = run_evaluation(limit=limit)
    print_results_table(results)

def main():
    parser = argparse.ArgumentParser(
        description="AskPlayStation AI Customer Support Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py run --text "my psn account got banned for no reason"
  python cli.py classify --text "error code ws-37397-9"
  python cli.py route --text "i want a refund for my game"
  python cli.py respond --text "how to restore licenses"
  python cli.py eval
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: run (full pipeline)
    parser_run = subparsers.add_parser("run", help="Execute full pipeline on input message")
    parser_run.add_argument("--text", type=str, required=True, help="Incoming customer query text")
    parser_run.add_argument("--context", type=str, default=None, help="Optional multi-turn thread context")
    parser_run.add_argument("-v", "--verbose", action="store_true", help="Print debug metadata and precedents")
    parser_run.set_defaults(func=cmd_run)

    # Subcommand: classify
    parser_classify = subparsers.add_parser("classify", help="Classify intent of customer message")
    parser_classify.add_argument("--text", type=str, required=True, help="Customer query text")
    parser_classify.add_argument("--context", type=str, default=None, help="Optional multi-turn context")
    parser_classify.set_defaults(func=cmd_classify)

    # Subcommand: respond
    parser_respond = subparsers.add_parser("respond", help="Draft grounded response")
    parser_respond.add_argument("--text", type=str, required=True, help="Customer query text")
    parser_respond.add_argument("--intent", type=str, default=None, help="Optional known intent")
    parser_respond.set_defaults(func=cmd_respond)

    # Subcommand: route
    parser_route = subparsers.add_parser("route", help="Evaluate routing decision (AUTO_HANDLE vs ESCALATE)")
    parser_route.add_argument("--text", type=str, required=True, help="Customer query text")
    parser_route.add_argument("--intent", type=str, default=None, help="Optional known intent")
    parser_route.add_argument("--confidence", type=float, default=0.85, help="Classification confidence")
    parser_route.set_defaults(func=cmd_route)

    # Subcommand: eval
    parser_eval = subparsers.add_parser("eval", help="Run benchmark evaluation over the golden set")
    parser_eval.add_argument("--limit", type=int, default=None, help="Optional limit on number of eval examples")
    parser_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)

if __name__ == "__main__":
    main()
