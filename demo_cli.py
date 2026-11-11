"""Preview a seeded ticket with offline rules or a locally installed Ollama model.

Usage: python3 demo_cli.py --ticket REQ-005
"""
import argparse
import csv
import json
from flowpilot.agent import analyze
from flowpilot.ollama import DEFAULT_MODEL
from flowpilot.storage import SAMPLES


def main():
    parser = argparse.ArgumentParser(description='Preview FlowPilot triage')
    parser.add_argument('--ticket', default='REQ-005', help='Sample ticket ID')
    parser.add_argument('--engine', choices=('offline', 'ollama'), default='offline')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()
    with SAMPLES.open(newline='', encoding='utf-8') as f:
        match = next((r for r in csv.DictReader(f) if r['id'] == args.ticket), None)
    if match is None:
        parser.error('Unknown sample ticket ID')
    result, sources = analyze(match['subject'], match['description'],
                              use_llm=args.engine == 'ollama', provider=args.engine, model=args.model)
    print(json.dumps({
        'ticket_id': args.ticket,
        'subject': match['subject'],
        'analysis': result.to_dict(),
        'retrieved_sources': [{'id': p['id'], 'title': p['title']} for p in sources],
    }, indent=2))
    return 1 if result.fallback else 0


if __name__ == '__main__':
    raise SystemExit(main())
