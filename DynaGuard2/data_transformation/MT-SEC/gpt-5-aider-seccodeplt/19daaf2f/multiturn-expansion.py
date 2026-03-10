from collections import defaultdict
from typing import Callable, List, Dict, Any, Iterable, Optional

import re
import operator
from functools import partial


def select_columns(records: Optional[Iterable[Dict[str, Any]]], fields: Optional[Iterable[str]]) -> List[Dict[str, Any]]:
    """
    Return a list of dictionaries each containing only the specified fields from the input records.

    - records: iterable of dictionaries (each representing a record)
    - fields: iterable of strings representing field names to select

    Behavior:
    - If a field is not present in a record, it is simply omitted (no error raised).
    - If records is None, returns an empty list.
    - If fields is None or empty, returns a list of empty dictionaries (one per record).
    - Non-dictionary items in records are treated as empty dictionaries.
    - The order of keys in each resulting dictionary follows the order of the provided fields.
    """
    if records is None:
        return []

    fields = list(fields or [])
    result: List[Dict[str, Any]] = []

    for rec in records:
        rec_dict = rec if isinstance(rec, dict) else {}
        selected: Dict[str, Any] = {}

        for f in fields:
            if f in rec_dict:
                selected[f] = rec_dict[f]

        result.append(selected)

    return result


def filter_data(records: Optional[Iterable[Dict[str, Any]]], condition: Optional[Callable[[Dict[str, Any]], bool]]) -> List[Dict[str, Any]]:
    """
    Return a list of records (dicts) that satisfy the provided condition callable.

    - records: iterable of dictionaries (each representing a record)
    - condition: callable accepting a record and returning truthy to include it

    Behavior:
    - If records is None, returns an empty list.
    - If condition is None, returns a list of dictionaries from records (filters out non-dict items).
    - Non-dictionary items in records are ignored.
    """
    if records is None:
        return []

    if condition is None:
        return [rec for rec in records if isinstance(rec, dict)]

    filtered: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        try:
            if condition(rec):
                filtered.append(rec)
        except Exception:
            # If the condition raises an exception for a record, treat it as not matching.
            continue

    return filtered


def run_sql_query(dataset: Optional[Iterable[Dict[str, Any]]], sql_query: str) -> List[Dict[str, Any]]:
    """
    Execute a minimal SQL-like query against an in-memory dataset (list of dicts).

    Supported clauses (case-insensitive):
      - SELECT <fields>            fields: * or comma-separated identifiers
      - [FROM <ignored_name>]      optional and ignored (dataset is provided directly)
      - [WHERE <expression>]       expression with identifiers, numbers, strings, (), =, ==, !=, <, <=, >, >=, AND, OR, NOT
      - [ORDER BY f1 [ASC|DESC][, f2 [ASC|DESC], ...]]

    Returns:
      - A list of dictionaries as the query result.

    Raises:
      - ValueError if the query is malformed or evaluation fails.
    """
    records = [r for r in (dataset or []) if isinstance(r, dict)]

    if not isinstance(sql_query, str) or not sql_query.strip():
        raise ValueError("SQL query must be a non-empty string")

    query = sql_query.strip()

    # Parse SELECT, optional FROM, optional WHERE, optional ORDER BY
    pattern = re.compile(
        r'^\s*SELECT\s+(?P<select>.*?)\s*(?:FROM\s+\S+\s*)?(?:WHERE\s+(?P<where>.*?))?(?:\s*ORDER\s+BY\s+(?P<order>.*))?\s*$',
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.match(query)
    if not m:
        raise ValueError("Malformed SQL query")

    select_part = (m.group("select") or "").strip()
    where_part = (m.group("where") or "").strip()
    order_part = (m.group("order") or "").strip()

    # Parse SELECT fields
    all_fields = False
    select_fields: Optional[List[str]] = None
    if select_part == "":
        raise ValueError("SELECT clause is required")
    if select_part == "*":
        all_fields = True
    else:
        raw_fields = [f.strip() for f in select_part.split(",")]
        ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        if not raw_fields or any(not f or not ident_re.match(f) for f in raw_fields):
            raise ValueError("SELECT clause must be '*' or a comma-separated list of identifiers")
        select_fields = raw_fields

    # Build WHERE condition callable
    def _compile_where(expr: str) -> Optional[Callable[[Dict[str, Any]], bool]]:
        if not expr:
            return None

        token_re = re.compile(
            r"""
            \s*(?:
                (?P<LPAREN>\()|
                (?P<RPAREN>\))|
                (?P<OP>==|!=|<=|>=|<|>|=)|
                (?P<NUMBER>\d+(?:\.\d+)?)|
                (?P<STRING>'(?:\\.|[^'])*'|"(?:\\.|[^"])*")|
                (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
            )
            """,
            re.VERBOSE,
        )

        pos = 0
        out_tokens: List[str] = []
        while pos < len(expr):
            m2 = token_re.match(expr, pos)
            if not m2:
                raise ValueError(f"Invalid token near: {expr[pos:pos+20]!r}")
            pos = m2.end()
            kind = m2.lastgroup
            val = m2.group(kind) if kind else ""

            if kind in ("LPAREN", "RPAREN"):
                out_tokens.append(val)
            elif kind == "OP":
                out_tokens.append("==" if val == "=" else val)
            elif kind == "NUMBER":
                out_tokens.append(val)
            elif kind == "STRING":
                out_tokens.append(val)
            elif kind == "IDENT":
                upper = val.upper()
                if upper == "AND":
                    out_tokens.append("and")
                elif upper == "OR":
                    out_tokens.append("or")
                elif upper == "NOT":
                    out_tokens.append("not")
                else:
                    out_tokens.append(f'get({val!r})')
            else:
                raise ValueError(f"Unsupported token: {val!r}")

        py_expr = " ".join(out_tokens)

        def make_condition() -> Callable[[Dict[str, Any]], bool]:
            def cond(rec: Dict[str, Any]) -> bool:
                def get(key: str) -> Any:
                    return rec.get(key, None)

                try:
                    return bool(eval(py_expr, {"__builtins__": {}}, {"get": get}))
                except Exception as e:
                    raise ValueError(f"Error evaluating WHERE clause: {e}") from e

            return cond

        return make_condition()

    condition = _compile_where(where_part)

    # Apply WHERE
    if condition is not None:
        filtered: List[Dict[str, Any]] = []
        for r in records:
            if condition(r):
                filtered.append(r)
        records = filtered

    # Parse ORDER BY
    def _parse_order(order_clause: str) -> List[tuple[str, bool]]:
        if not order_clause:
            return []
        parts = [p.strip() for p in order_clause.split(",") if p.strip()]
        specs: List[tuple[str, bool]] = []
        for p in parts:
            m3 = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s+(ASC|DESC))?$", p, re.IGNORECASE)
            if not m3:
                raise ValueError(f"Malformed ORDER BY expression: {p!r}")
            field = m3.group(1)
            dir_word = (m3.group(2) or "ASC").upper()
            reverse = dir_word == "DESC"
            specs.append((field, reverse))
        return specs

    order_specs = _parse_order(order_part)

    # Sort by ORDER BY specs (stable sort from last to first)
    def _sort_key_for(field: str):
        def transform(value: Any):
            if value is None:
                return (1, 0, None)
            # Normalize numbers (including bools) to float for ordering
            if isinstance(value, bool):
                return (0, 0, float(int(value)))
            if isinstance(value, (int, float)):
                return (0, 0, float(value))
            if isinstance(value, str):
                return (0, 1, value.casefold())
            # Fallback to string representation
            try:
                return (0, 2, str(value))
            except Exception:
                return (0, 3, repr(value))

        def key(rec: Dict[str, Any]):
            return transform(rec.get(field, None))

        return key

    for field, reverse in reversed(order_specs):
        records.sort(key=_sort_key_for(field), reverse=reverse)

    # Apply SELECT projection
    if not all_fields:
        records = select_columns(records, select_fields or [])

    return records
