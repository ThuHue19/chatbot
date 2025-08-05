# File: utils/column_mapper.py (new)
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEYWORDS_FILE = os.path.join(BASE_DIR, "table_keywords.json")

with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
    TABLE_KEYWORDS = json.load(f)

def extract_tables_and_columns(question: str):
    question = question.lower()
    matched_tables = set()
    matched_columns = {}

    for table, info in TABLE_KEYWORDS.items():
        table_keywords = info.get("keywords", [])
        if any(kw.lower() in question for kw in table_keywords):
            matched_tables.add(table)

        for col, col_info in info.get("columns", {}).items():
            for kw in col_info.get("keywords", []):
                if kw.lower() in question:
                    matched_tables.add(table)
                    matched_columns.setdefault(table, set()).add(col)

    return list(matched_tables), matched_columns

def generate_column_mapping_hint(question):
    question = question.lower()
    hints = []
    for table, info in TABLE_KEYWORDS.items():
        for col, col_info in info.get("columns", {}).items():
            for kw in col_info.get("keywords", []):
                if kw.lower() in question:
                    hints.append(f"- '{kw}' → {table}.{col}")
    return "\n".join(hints)
