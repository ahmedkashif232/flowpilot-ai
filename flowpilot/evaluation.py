"""Compare triage engines and policy retrieval on versioned synthetic cases."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

from . import ollama
from .agent import CATEGORIES, analyze
from .policies import POLICIES_PATH, ROOT

CASES = ROOT / 'data' / 'evaluation_cases.csv'


def classification_metrics(rows):
    by_category = {}
    for category in CATEGORIES:
        actual = sum(row['Expected category'] == category for row in rows)
        predicted = sum(row['Predicted category'] == category for row in rows)
        correct = sum(row['Expected category'] == category == row['Predicted category'] for row in rows)
        by_category[category] = {
            'Support': actual,
            'Precision': correct / predicted if predicted else 0.0,
            'Recall': correct / actual if actual else 0.0,
            'F1': 2 * correct / (actual + predicted) if actual + predicted else 0.0,
        }
    supported = [entry['F1'] for entry in by_category.values() if entry['Support']]
    return {
        'cases': len(rows),
        'category_accuracy': sum(row['Category correct'] for row in rows) / len(rows) if rows else None,
        'priority_accuracy': sum(row['Priority correct'] for row in rows) / len(rows) if rows else None,
        'macro_f1': sum(supported) / len(supported) if supported else None,
        'by_category': by_category,
    }


def run_evaluation(path=CASES, engine='offline', model=ollama.DEFAULT_MODEL):
    if engine not in ('offline', 'ollama'):
        raise ValueError('Choose offline or ollama.')
    model_info = ollama.installed_model(model) if engine == 'ollama' else {}
    with path.open(newline='', encoding='utf-8') as file:
        cases = list(csv.DictReader(file))
    rows = []
    for case in cases:
        start = perf_counter()
        result, sources = analyze(case['subject'], case['description'],
                                  use_llm=engine == 'ollama', provider=engine, model=model)
        elapsed = perf_counter() - start
        retrieved = [source['id'] for source in sources]
        target = case.get('expected_policy_id') or None
        rows.append({
            'Request': case['id'], 'Subject': case['subject'],
            'Expected category': case['expected_category'], 'Predicted category': result.category,
            'Category correct': result.category == case['expected_category'],
            'Expected priority': case['expected_priority'], 'Predicted priority': result.priority,
            'Priority correct': result.priority == case['expected_priority'],
            'Expected policy': target, 'Retrieved': ', '.join(retrieved),
            'Citations': ', '.join(result.citation_ids),
            'Policy retrieved': target in retrieved if target else None,
            'Policy cited': target in result.citation_ids if target else None,
            'Engine': result.engine, 'Fallback': result.fallback,
            'Latency seconds': round(elapsed, 4), 'Warning': result.warning,
        })
    completed = [row for row in rows if row['Engine'] == engine]
    labeled = [row for row in rows if row['Expected policy']]
    return {
        'engine': engine, 'model': model if model_info else None,
        'model_digest': model_info.get('digest'),
        'dataset': path.name, 'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'policies_sha256': hashlib.sha256(POLICIES_PATH.read_bytes()).hexdigest(),
        'machine': f'{platform.system()} {platform.release()} {platform.machine()}',
        **classification_metrics(rows),
        'engine_only': classification_metrics(completed),
        'fallback_count': len(rows) - len(completed),
        'policy_hit_at_3': sum(row['Policy retrieved'] for row in labeled) / len(labeled) if labeled else None,
        'target_citation_rate': sum(row['Policy cited'] for row in labeled) / len(labeled) if labeled else None,
        'median_seconds': median(row['Latency seconds'] for row in rows) if rows else None,
        'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('offline', 'ollama', 'both'), default='offline')
    parser.add_argument('--model', default=ollama.DEFAULT_MODEL)
    parser.add_argument('--cases', type=Path, default=CASES)
    parser.add_argument('--output', type=Path, help='Write JSON results to this file')
    args = parser.parse_args()
    engines = ('offline', 'ollama') if args.engine == 'both' else (args.engine,)
    try:
        reports = {engine: run_evaluation(args.cases, engine, args.model) for engine in engines}
    except (ollama.OllamaError, ValueError, OSError) as exc:
        parser.exit(1, f'Evaluation failed. {exc}\n')
    output = json.dumps(reports, indent=2)
    if args.output:
        args.output.write_text(output + '\n', encoding='utf-8')
    else:
        print(output)


if __name__ == '__main__':
    main()
