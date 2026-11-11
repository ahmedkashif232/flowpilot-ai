"""SQLite-backed sample tickets and human approval audit records."""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / 'data' / 'flowpilot.db'
SAMPLES = ROOT / 'data' / 'sample_requests.csv'


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def db_path() -> Path:
    return Path(os.getenv('FLOWPILOT_DB', str(DEFAULT_DB)))


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys=ON')
    return conn


def init_db(path: Path | None = None, samples: Path = SAMPLES) -> None:
    with closing(connect(path)) as conn, conn:
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY,
            requester TEXT NOT NULL,
            subject TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analyses (
            request_id TEXT PRIMARY KEY REFERENCES requests(id),
            result_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('Pending review', 'Approved', 'Rejected')),
            reviewed_reply TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL REFERENCES requests(id),
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            details TEXT NOT NULL,
            at TEXT NOT NULL
        );
        ''')
        conn.execute('BEGIN IMMEDIATE')
        count = conn.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
        if count == 0 and samples.exists():
            with samples.open(newline='', encoding='utf-8') as f:
                for item in csv.DictReader(f):
                    conn.execute('INSERT INTO requests VALUES (?,?,?,?,?)',
                                 (item['id'], item['requester'], item['subject'], item['description'], now()))


def _request_values(requester, subject, description):
    if not subject.strip() or not description.strip():
        raise ValueError('Subject and description are required.')
    if len(description) > 12000 or len(subject) > 300 or len(requester) > 100:
        raise ValueError('Input exceeds the request length limits.')
    return (
        f"REQ-{uuid.uuid4().hex[:12].upper()}", requester.strip() or 'Unknown',
        subject.strip(), description.strip(), now(),
    )


def _insert_request(conn, values, source):
    conn.execute('INSERT INTO requests VALUES (?,?,?,?,?)', values)
    conn.execute('INSERT INTO events(request_id,actor,action,details,at) VALUES (?,?,?,?,?)',
                 (values[0], 'User', 'Created request', source, now()))


def add_request(requester: str, subject: str, description: str, path: Path | None = None) -> str:
    values = _request_values(requester, subject, description)
    with closing(connect(path)) as conn, conn:
        _insert_request(conn, values, 'Manual entry')
    return values[0]


def import_csv(contents: bytes, path: Path | None = None) -> int:
    if len(contents) > 2 * 1024 * 1024:
        raise ValueError('CSV files must be 2 MB or smaller.')
    rows = csv.DictReader(io.StringIO(contents.decode('utf-8-sig'), newline=''), strict=True)
    fields = rows.fieldnames or []
    if len(fields) != len(set(fields)) or not {'subject', 'description'}.issubset(fields):
        raise ValueError('CSV must include unique subject and description columns.')
    batch = []
    for number, row in enumerate(rows, start=1):
        if number > 100:
            raise ValueError('Maximum 100 requests per upload.')
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Column count does not match the header.')
            batch.append(_request_values(row.get('requester', ''), row['subject'], row['description']))
        except ValueError as exc:
            raise ValueError(f'CSV record {number}: {exc}') from exc
    if not batch:
        raise ValueError('CSV contains no requests.')
    with closing(connect(path)) as conn, conn:
        for values in batch:
            _insert_request(conn, values, 'CSV import')
    return len(batch)


def list_requests(path: Path | None = None) -> list[dict[str, Any]]:
    with closing(connect(path)) as conn, conn:
        rows = conn.execute('''SELECT r.*, COALESCE(a.status, 'New') AS status,
                                a.result_json, a.reviewed_reply, a.updated_at AS revision
                                FROM requests r LEFT JOIN analyses a ON r.id=a.request_id
                                ORDER BY r.created_at, r.id''').fetchall()
    result = [dict(row) for row in rows]
    for row in result:
        row['analysis'] = json.loads(row.pop('result_json')) if row['result_json'] else None
    return result


def save_analysis(request_id: str, result: dict[str, Any], path: Path | None = None) -> None:
    with closing(connect(path)) as conn, conn:
        conn.execute('''INSERT INTO analyses VALUES (?, ?, 'Pending review', NULL, ?)
                     ON CONFLICT(request_id) DO UPDATE SET result_json=excluded.result_json,
                     status=excluded.status, reviewed_reply=NULL, updated_at=excluded.updated_at''',
                     (request_id, json.dumps(result), now()))
        conn.execute('INSERT INTO events(request_id,actor,action,details,at) VALUES (?,?,?,?,?)',
                     (request_id, 'Analysis engine', 'Generated recommendation', result.get('mode', ''), now()))


def record_decision(request_id: str, decision: str, edited_reply: str, reviewer: str = 'Human reviewer', path: Path | None = None,
                    *, expected_revision: str | None = None) -> None:
    if decision not in ('Approved', 'Rejected'):
        raise ValueError('Invalid decision')
    if len(edited_reply) > 5000:
        raise ValueError('Reply too long')
    with closing(connect(path)) as conn, conn:
        revision_clause = ' AND updated_at=?' if expected_revision is not None else ''
        params = (decision, edited_reply if decision == 'Approved' else '', now(), request_id)
        if expected_revision is not None:
            params += (expected_revision,)
        changed = conn.execute(
            "UPDATE analyses SET status=?, reviewed_reply=?, updated_at=? "
            "WHERE request_id=? AND status='Pending review'" + revision_clause, params
        )
        if changed.rowcount != 1:
            raise ValueError('This proposal changed or was already reviewed. Reload and review the latest draft.')
        conn.execute('INSERT INTO events(request_id,actor,action,details,at) VALUES (?,?,?,?,?)',
                     (request_id, reviewer, decision, 'Simulation only: no external action or email was sent.', now()))


def list_events(path: Path | None = None) -> list[dict[str, Any]]:
    with closing(connect(path)) as conn, conn:
        return [dict(row) for row in conn.execute('SELECT * FROM events ORDER BY id DESC LIMIT 200').fetchall()]
