import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.utils import logger, get_project_root

DB_PATH = get_project_root() / "bugs.db"

def get_connection():
    """Establishes connection to SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates bug_reports table if it does not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bug_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            user TEXT DEFAULT 'Developer',
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            module TEXT NOT NULL,
            team TEXT NOT NULL,
            causes TEXT NOT NULL,
            tests TEXT NOT NULL,
            confidence REAL NOT NULL,
            explanation TEXT NOT NULL,
            tokens TEXT NOT NULL,
            keyphrases TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Initialized SQLite database at {DB_PATH}")

def save_bug_analysis(record: Dict[str, Any]) -> int:
    """Saves bug analysis record into SQLite database."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    inp = record.get("input", {})
    pred = record.get("prediction", {})
    nlp = record.get("nlp", {})

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y-%m-%d")

    cursor.execute("""
        INSERT INTO bug_reports (
            title, description, user, date, category, severity, module, team,
            causes, tests, confidence, explanation, tokens, keyphrases, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        inp.get("title", ""),
        inp.get("description", ""),
        record.get("user", "Senior Developer"),
        date_str,
        pred.get("category", "Logic Error"),
        pred.get("severity", "Medium"),
        pred.get("affected_module", "General Component"),
        pred.get("suggested_team", "Backend Services"),
        json.dumps(pred.get("possible_causes", [])),
        json.dumps(pred.get("recommended_tests", [])),
        float(pred.get("confidence_score", 0.85)),
        pred.get("explanation", ""),
        json.dumps(nlp.get("tokens", [])),
        json.dumps(nlp.get("keyphrases", [])),
        now_str
    ))

    bug_id = cursor.lastrowid
    conn.commit()
    conn.close()

    logger.info(f"Persisted bug analysis record with ID #{bug_id} to database.")
    return bug_id

def get_all_analyses() -> List[Dict[str, Any]]:
    """Fetches all stored bug analyses ordered by newest first."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM bug_reports ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    records = []
    for r in rows:
        rec = dict(r)
        rec["causes"] = json.loads(rec["causes"]) if rec["causes"] else []
        rec["tests"] = json.loads(rec["tests"]) if rec["tests"] else []
        rec["tokens"] = json.loads(rec["tokens"]) if rec["tokens"] else []
        rec["keyphrases"] = json.loads(rec["keyphrases"]) if rec["keyphrases"] else []
        records.append(rec)

    return records

def get_bug_by_id(bug_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single bug analysis by ID."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM bug_reports WHERE id = ?", (bug_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    rec = dict(row)
    rec["causes"] = json.loads(rec["causes"]) if rec["causes"] else []
    rec["tests"] = json.loads(rec["tests"]) if rec["tests"] else []
    rec["tokens"] = json.loads(rec["tokens"]) if rec["tokens"] else []
    rec["keyphrases"] = json.loads(rec["keyphrases"]) if rec["keyphrases"] else []
    return rec

def search_bugs(query: str) -> List[Dict[str, Any]]:
    """Performs LIKE search over title, description, category, and module."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    q = f"%{query}%"
    cursor.execute("""
        SELECT * FROM bug_reports
        WHERE title LIKE ? OR description LIKE ? OR category LIKE ? OR module LIKE ? OR team LIKE ?
        ORDER BY id DESC
    """, (q, q, q, q, q))

    rows = cursor.fetchall()
    conn.close()

    records = []
    for r in rows:
        rec = dict(r)
        rec["causes"] = json.loads(rec["causes"]) if rec["causes"] else []
        rec["tests"] = json.loads(rec["tests"]) if rec["tests"] else []
        rec["tokens"] = json.loads(rec["tokens"]) if rec["tokens"] else []
        rec["keyphrases"] = json.loads(rec["keyphrases"]) if rec["keyphrases"] else []
        records.append(rec)

    return records

def get_stats() -> Dict[str, Any]:
    """Computes summary statistics for analytics dashboard."""
    records = get_all_analyses()
    if not records:
        return {
            "total_bugs": 0,
            "most_common_category": "N/A",
            "critical_count": 0,
            "recent_count": 0
        }

    total = len(records)
    categories = [r["category"] for r in records]
    from collections import Counter
    cat_counts = Counter(categories)
    most_common = cat_counts.most_common(1)[0][0] if cat_counts else "N/A"
    critical_count = sum(1 for r in records if r["severity"].lower() == "critical")

    return {
        "total_bugs": total,
        "most_common_category": most_common,
        "critical_count": critical_count,
        "recent_count": min(total, 5)
    }
