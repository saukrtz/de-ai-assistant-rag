"""
app/agents/tools/catalog_explorer.py
──────────────────────────────────────
Data Catalogue Explorer Tool.

Logical flow:
  1. Load tables.json and lineage.json from data/catalogue/.
  2. Support search by: table name, column name, PII tag, owner.
  3. Support lineage traversal: upstream and downstream tables.
  4. Return structured results for display in the UI.
"""

import json
import logging
from pathlib import Path
from app.config import settings

logger = logging.getLogger(__name__)

_tables: list[dict] | None = None
_lineage: dict | None = None


def _load_data():
    global _tables, _lineage
    if _tables is None:
        tables_path = settings.catalogue_dir / "tables.json"
        lineage_path = settings.catalogue_dir / "lineage.json"
        _tables = json.loads(tables_path.read_text())
        _lineage = json.loads(lineage_path.read_text())


def search_tables(query: str) -> list[dict]:
    """
    Search tables by name, column, PII tag, or owner.
    Returns matching table records.
    """
    _load_data()
    query_lower = query.lower()
    results = []
    for table in _tables:
        name_match = query_lower in table.get("name", "").lower()
        owner_match = query_lower in table.get("owner", "").lower()
        pii_match = any(
            query_lower in tag.lower() for tag in table.get("pii_tags", [])
        )
        col_match = any(
            query_lower in col.get("name", "").lower()
            for col in table.get("columns", [])
        )
        if name_match or owner_match or pii_match or col_match:
            results.append(table)
    logger.info(f"Catalogue search '{query}': {len(results)} hits")
    return results


def get_lineage(table_name: str, direction: str = "both") -> dict:
    """
    Get upstream / downstream lineage for a table.

    Args:
        table_name: Table to query.
        direction: 'upstream', 'downstream', or 'both'.

    Returns:
        dict with 'upstream' and 'downstream' lists.
    """
    _load_data()
    edges = _lineage.get("edges", [])
    upstream = [
        e["source"] for e in edges if e["target"] == table_name
    ]
    downstream = [
        e["target"] for e in edges if e["source"] == table_name
    ]
    result = {}
    if direction in ("upstream", "both"):
        result["upstream"] = upstream
    if direction in ("downstream", "both"):
        result["downstream"] = downstream
    return result


def get_pii_tables() -> list[dict]:
    """Return all tables that contain PII columns."""
    _load_data()
    return [t for t in _tables if t.get("pii_tags")]


def get_table_by_name(name: str) -> dict | None:
    """Return a single table definition by exact name."""
    _load_data()
    for t in _tables:
        if t["name"].lower() == name.lower():
            return t
    return None
