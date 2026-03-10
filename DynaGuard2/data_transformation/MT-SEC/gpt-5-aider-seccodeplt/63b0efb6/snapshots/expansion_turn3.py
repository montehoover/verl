import re
import operator
from functools import partial
from collections import defaultdict
from typing import Callable, List, Dict

__all__ = ["select_fields", "filter_data", "run_sql_query"]


def select_fields(records, fields):
    """
    Extract specified fields from a list of record dictionaries.

    - records: list of dict objects (each record).
    - fields: list of strings specifying field names to extract.

    Returns a new list of dicts containing only the requested fields.
    Missing fields in a record are included with value None.
    """
    if records is None:
        return []

    if not isinstance(fields, (list, tuple)):
        raise TypeError("fields must be a list or tuple of field names")

    result = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each record must be a dict")
        rec_with_default = defaultdict(lambda: None, record)
        result.append({field: rec_with_default[field] for field in fields})
    return result


def filter_data(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filter records based on a provided condition callable.

    - records: list of dict objects (each record).
    - condition: callable that accepts a record dict and returns True if the
      record should be included.

    Returns a new list containing only the records that satisfy the condition.
    """
    if records is None:
        return []

    if not callable(condition):
        raise TypeError("condition must be a callable that accepts a record and returns a bool")

    filtered = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("each record must be a dict")
        if condition(record):
            filtered.append(record)
    return filtered


def run_sql_query(records: List[Dict], command: str) -> List[Dict]:
    """
    Execute a simple SQL-like query on a list of record dictionaries.

    Supported syntax (case-insensitive):
      SELECT <fields | *>
      [WHERE <field op value> [AND <field op value> ...] [OR <...>] ...]
      [ORDER BY <field> [ASC|DESC] [, <field> [ASC|DESC] ...]]

    - records: list of dicts.
    - command: SQL-like string.

    Returns a list of dicts representing the query result.
    Raises ValueError if the query is malformed or cannot be evaluated.
    """
    if records is None:
        records = []
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("records must be a list of dictionaries")
    if not isinstance(command, str):
        raise ValueError("command must be a string")

    cmd = command.strip()
    if not cmd:
        raise ValueError("empty command")

    # Parse SELECT / WHERE / ORDER BY
    m = re.match(
        r"^\s*SELECT\s+(?P<select>.+?)(?:\s+WHERE\s+(?P<where>.+?))?(?:\s+ORDER\s+BY\s+(?P<order>.+))?\s*$",
        cmd,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        raise ValueError("Malformed query: could not parse SELECT/WHERE/ORDER BY")

    select_clause = m.group("select").strip()
    where_clause = m.group("where")
    order_clause = m.group("order")

    # Parse fields
    select_all = select_clause == "*"
    if not select_all:
        # comma-separated identifiers
        fields = [f.strip() for f in select_clause.split(",")]
        if not fields or any(not re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*$", f) for f in fields):
            raise ValueError("Malformed SELECT fields")
    else:
        fields = None  # indicates select all

    # Helpers for WHERE parsing
    op_map = {
        "=": operator.eq,
        "==": operator.eq,
        "!=": operator.ne,
        "<>": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }

    ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.]*$")

    def parse_value(token: str):
        t = token.strip()
        # Quoted strings
        if (len(t) >= 2) and ((t[0] == t[-1] == "'") or (t[0] == t[-1] == '"')):
            inner = t[1:-1]
            inner = inner.replace(r"\\", "\\").replace(r"\'", "'").replace(r"\"", '"')
            return inner, False  # constant
        # NULL
        if t.upper() == "NULL":
            return None, False
        # Booleans
        if t.upper() == "TRUE":
            return True, False
        if t.upper() == "FALSE":
            return False, False
        # Numeric
        if re.match(r"^[+-]?\d+(\.\d+)?$", t):
            if "." in t:
                try:
                    return float(t), False
                except Exception as e:
                    raise ValueError(f"Invalid numeric literal: {t}") from e
            try:
                return int(t), False
            except Exception as e:
                raise ValueError(f"Invalid integer literal: {t}") from e
        # Identifier (field reference)
        if ident_re.match(t):
            return t, True  # field ref
        raise ValueError(f"Invalid value token: {t}")

    def parse_simple_condition(expr: str):
        # field op value
        mm = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(=|==|!=|<>|<=|<|>=|>)\s*(.+?)\s*$", expr)
        if not mm:
            raise ValueError(f"Malformed condition: {expr}")
        left_field = mm.group(1)
        op_str = mm.group(2)
        right_token = mm.group(3)
        right_parsed, is_field = parse_value(right_token)
        if op_str not in op_map:
            raise ValueError(f"Unsupported operator: {op_str}")
        op_func = op_map[op_str]

        def cond(record: Dict) -> bool:
            try:
                left_val = record.get(left_field, None)
                right_val = record.get(right_parsed, None) if is_field else right_parsed
                return op_func(left_val, right_val)
            except Exception as e:
                raise ValueError(f"Failed to evaluate condition '{expr}': {e}") from e

        return cond

    # Build predicate from WHERE clause with AND having higher precedence than OR
    def build_predicate(where_text: str):
        if where_text is None or not where_text.strip():
            return lambda _r: True

        # Split by OR (case-insensitive)
        or_terms = re.split(r"\s+OR\s+", where_text, flags=re.IGNORECASE)
        terms = []
        for term in or_terms:
            and_parts = re.split(r"\s+AND\s+", term, flags=re.IGNORECASE)
            if not and_parts:
                raise ValueError(f"Malformed WHERE term: {term}")
            conds = [parse_simple_condition(p.strip()) for p in and_parts if p.strip()]
            if not conds:
                raise ValueError(f"Malformed WHERE term: {term}")
            terms.append(conds)

        def predicate(record: Dict) -> bool:
            # Any AND-clause group satisfied
            for conds in terms:
                ok = True
                for c in conds:
                    if not c(record):
                        ok = False
                        break
                if ok:
                    return True
            return False

        return predicate

    predicate = build_predicate(where_clause)

    # Filter
    try:
        filtered_records = [r for r in records if predicate(r)]
    except ValueError:
        # propagate ValueError from predicate
        raise
    except Exception as e:
        raise ValueError(f"Failed to evaluate WHERE clause: {e}") from e

    # Order By
    def safe_key(val):
        # Ensure comparability across mixed types and None
        if val is None:
            return (1, 0, None)
        if isinstance(val, bool):
            return (0, 0, int(val))
        if isinstance(val, (int, float)):
            return (0, 1, val)
        if isinstance(val, str):
            return (0, 2, val)
        return (0, 3, str(val))

    if order_clause:
        try:
            specs = [s.strip() for s in order_clause.split(",") if s.strip()]
            order_specs = []
            for spec in specs:
                parts = spec.split()
                if not parts:
                    raise ValueError(f"Malformed ORDER BY part: '{spec}'")
                field = parts[0]
                direction = "ASC"
                if len(parts) >= 2:
                    dir_token = parts[1].upper()
                    if dir_token not in ("ASC", "DESC"):
                        raise ValueError(f"Invalid ORDER BY direction: {parts[1]}")
                    direction = dir_token
                order_specs.append((field, direction))

            # Stable sorts from last to first
            ordered = list(filtered_records)
            for field, direction in reversed(order_specs):
                ordered.sort(key=lambda r, f=field: safe_key(r.get(f, None)), reverse=(direction == "DESC"))
            filtered_records = ordered
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to evaluate ORDER BY: {e}") from e

    # Projection (SELECT)
    try:
        if select_all:
            # Return shallow copies to avoid mutating original records
            return [dict(r) for r in filtered_records]
        return select_fields(filtered_records, fields)
    except Exception as e:
        raise ValueError(f"Failed to evaluate SELECT clause: {e}") from e
