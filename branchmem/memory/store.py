"""SQLite-backed fact store with branch and provenance tracking."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from branchmem.memory.schemas import MemoryBranch, MemoryFact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS branches (
    branch_id TEXT PRIMARY KEY,
    parent_branch_id TEXT,
    fork_point_timestamp REAL
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    entity TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    branch_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    source TEXT NOT NULL,
    confidence REAL NOT NULL,
    provenance TEXT NOT NULL,
    common_ancestor_id TEXT,
    FOREIGN KEY (branch_id) REFERENCES branches (branch_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_branch ON facts (branch_id);
CREATE INDEX IF NOT EXISTS idx_facts_entity_predicate ON facts (entity, predicate);
"""


class MemoryStore:
    """Thin SQLite persistence layer for MemoryBranch / MemoryFact objects.

    Uses ':memory:' by default for tests; pass a file path for a persistent store.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- branches ---------------------------------------------------------

    def create_branch(self, branch: MemoryBranch) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO branches (branch_id, parent_branch_id, "
            "fork_point_timestamp) VALUES (?, ?, ?)",
            (branch.branch_id, branch.parent_branch_id, branch.fork_point_timestamp),
        )
        self._conn.commit()
        for fact in branch.facts:
            self.add_fact(fact)

    def get_branch(self, branch_id: str) -> Optional[MemoryBranch]:
        row = self._conn.execute(
            "SELECT * FROM branches WHERE branch_id = ?", (branch_id,)
        ).fetchone()
        if row is None:
            return None
        facts = self.get_branch_facts(branch_id)
        return MemoryBranch(
            branch_id=row["branch_id"],
            parent_branch_id=row["parent_branch_id"],
            fork_point_timestamp=row["fork_point_timestamp"],
            facts=facts,
        )

    # -- facts --------------------------------------------------------------

    def add_fact(self, fact: MemoryFact) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO facts (fact_id, entity, predicate, value, "
            "branch_id, timestamp, source, confidence, provenance, common_ancestor_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact.fact_id,
                fact.entity,
                fact.predicate,
                fact.value,
                fact.branch_id,
                fact.timestamp,
                fact.source,
                fact.confidence,
                fact.provenance,
                fact.common_ancestor_id,
            ),
        )
        self._conn.commit()

    def get_fact(self, fact_id: str) -> Optional[MemoryFact]:
        row = self._conn.execute(
            "SELECT * FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        return _row_to_fact(row) if row else None

    def get_branch_facts(self, branch_id: str) -> list[MemoryFact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE branch_id = ? ORDER BY timestamp", (branch_id,)
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def get_facts_by_key(self, branch_id: str, entity: str, predicate: str) -> list[MemoryFact]:
        rows = self._conn.execute(
            "SELECT * FROM facts WHERE branch_id = ? AND entity = ? AND predicate = ? "
            "ORDER BY timestamp",
            (branch_id, entity, predicate),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def all_facts(self) -> list[MemoryFact]:
        rows = self._conn.execute("SELECT * FROM facts ORDER BY timestamp").fetchall()
        return [_row_to_fact(r) for r in rows]


def _row_to_fact(row: sqlite3.Row) -> MemoryFact:
    return MemoryFact(
        fact_id=row["fact_id"],
        entity=row["entity"],
        predicate=row["predicate"],
        value=row["value"],
        branch_id=row["branch_id"],
        timestamp=row["timestamp"],
        source=row["source"],
        confidence=row["confidence"],
        provenance=row["provenance"],
        common_ancestor_id=row["common_ancestor_id"],
    )


def dump_branch_json(branch: MemoryBranch) -> str:
    return json.dumps(branch.model_dump(), indent=2)
