"""Minimal transparent local retrieval over a fictional internal-policy knowledge base.

This is lexical RAG (retrieval + a contextual response), not embedding/vector RAG.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICIES_PATH = ROOT / "data" / "policies.json"
STOP = set("a an and are as at be by can do for from has have how i in is it me my of on or our please the them they this to us was we what when where will with you your new not".split())


def load_policies(path: Path = POLICIES_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def tokens(value: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in STOP and len(w) > 1]


def retrieve(query: str, policies: list[dict[str, Any]] | None = None, category: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
    """Rank source paragraphs with interpretable BM25-like weighted lexical similarity."""
    policies = policies if policies is not None else load_policies()
    if not policies or not query.strip():
        return []
    query_terms = Counter(tokens(query))
    documents = [tokens(p['title'] + ' ' + p['title'] + ' ' + p['text']) for p in policies]
    df = Counter(term for doc in documents for term in set(doc))
    n = len(policies)
    avglen = sum(map(len, documents)) / n or 1
    results = []
    for p, doc in zip(policies, documents):
        tf = Counter(doc)
        score = 0.0
        for term, query_weight in query_terms.items():
            if not tf[term]:
                continue
            idf = math.log(1 + (n - df[term] + .5) / (df[term] + .5))
            norm = tf[term] * 2.2 / (tf[term] + 1.2 * (.25 + .75 * len(doc) / avglen))
            score += idf * norm * min(query_weight, 2)
        if category and p['category'] == category and score > 0:
            score *= 1.22
        if score > 0:
            results.append({**p, "score": round(score, 3)})
    return sorted(results, key=lambda p: p["score"], reverse=True)[:limit]


def lookup_policy(policy_id: str, policies: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    return next((p for p in (policies if policies is not None else load_policies()) if p['id'] == policy_id), None)
