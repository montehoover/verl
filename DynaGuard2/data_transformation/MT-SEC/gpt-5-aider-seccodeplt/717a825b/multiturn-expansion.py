import re
import operator
from functools import partial
from collections import defaultdict
from typing import Any, Callable, Dict, List


def select_fields(records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """
    Extract a subset of fields from each record in a list of dictionaries.

    - records: list of dictionaries representing records
    - fields: list of field names to select from each record

    Returns a list of dictionaries containing only the specified fields.
    If a field is missing in a record, its value will be None.
    """
    if records is None:
        return []

    result: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            record = {}
        selected = defaultdict(lambda: None)
        for field in fields:
            selected[field] = record.get(field)
        result.append(dict(selected))

    return result


def filter_data(records: List[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Filter records by a user-provided condition.

    - records: list of dictionaries representing records
    - condition: a callable that takes a record (dict) and returns True if the record should be included.

    Returns a list of records for which condition(record) evaluates to True.
    """
    if records is None:
        return []
    return [record for record in records if condition(record)]


def execute_custom_query(data: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Execute a simple SQL-like query against a list of dictionaries.

    Supported syntax (case-insensitive keywords):
      SELECT <fields|*>
      [FROM <ignored>]
      [WHERE <boolean expression>]
      [ORDER BY <field> [ASC|DESC]]

    - data: list of dictionaries representing records
    - query: SQL-like string

    Returns a list of dictionaries (the result set).
    Raises ValueError on invalid queries or execution errors.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    data = data or []

    # Parse query into SELECT, WHERE, ORDER BY parts
    pattern = re.compile(
        r'^\s*SELECT\s+(?P<select>.+?)\s*(?:FROM\s+\S+)?\s*(?:WHERE\s+(?P<where>.+?))?\s*(?:ORDER\s+BY\s+(?P<order>.+?))?\s*$',
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.match(query)
    if not match:
        raise ValueError("Invalid query format.")

    select_part = match.group('select').strip()
    where_part = match.group('where')
    order_part = match.group('order')

    # Determine fields to select
    if select_part == '*':
        select_fields_list: List[str] = []
        select_all = True
    else:
        # Split by commas not inside quotes
        try:
            select_fields_list = _split_csv(select_part)
        except Exception as e:
            raise ValueError(f"Invalid SELECT fields: {e}") from e
        select_all = False

    # Build WHERE condition function, if any
    condition_func: Callable[[Dict[str, Any]], bool]
    if where_part:
        try:
            condition_func = _build_where_condition(where_part)
        except Exception as e:
            raise ValueError(f"Invalid WHERE clause: {e}") from e
    else:
        condition_func = lambda _rec: True  # include all

    # Apply filtering
    try:
        filtered = [rec for rec in data if condition_func(rec)]
    except Exception as e:
        raise ValueError(f"Error evaluating WHERE clause: {e}") from e

    # Apply ordering
    ordered = filtered
    if order_part and order_part.strip():
        try:
            order_fields = _parse_order_by(order_part)
            ordered = _apply_order_by(ordered, order_fields)
        except Exception as e:
            raise ValueError(f"Invalid ORDER BY clause: {e}") from e

    # Apply selection
    if select_all:
        result = [dict(rec) for rec in ordered]
    else:
        result = select_fields(ordered, select_fields_list)

    return result


def _split_csv(s: str) -> List[str]:
    """
    Split a comma-separated list of identifiers, ignoring commas inside quotes.
    Returns list of trimmed items.
    """
    items: List[str] = []
    current = []
    in_single = False
    in_double = False
    escape = False

    for ch in s:
        if escape:
            current.append(ch)
            escape = False
            continue
        if ch == '\\':
            current.append(ch)
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            continue
        if ch == ',' and not in_single and not in_double:
            item = ''.join(current).strip()
            if item:
                items.append(item)
            current = []
        else:
            current.append(ch)
    tail = ''.join(current).strip()
    if tail:
        items.append(tail)

    # Validate identifiers that are not quoted string literals
    normalized: List[str] = []
    for item in items:
        if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
            # Allow string literal in SELECT but this is uncommon; treat as literal alias-less column
            normalized.append(item)
        else:
            normalized.append(item.strip())
    return normalized


def _build_where_condition(where_clause: str) -> Callable[[Dict[str, Any]], bool]:
    """
    Build a callable taking a record and returning a boolean per WHERE expression.
    Transforms identifiers into get('field') lookups and normalizes operators.
    """
    expr = _transform_where_to_python(where_clause)

    try:
        code = compile(expr, '<where>', 'eval')
    except Exception as e:
        raise ValueError(f"Unable to compile WHERE expression: {e}") from e

    def condition(record: Dict[str, Any]) -> bool:
        def get(name: str, default: Any = None) -> Any:
            return record.get(name, default)

        # Disallow builtins; expose only 'get', True/False/None via literals
        env = {'__builtins__': {}}
        local_env = {'get': get}
        try:
            return bool(eval(code, env, local_env))
        except Exception as e:
            raise ValueError(f"Failed to evaluate WHERE expression on record {record}: {e}") from e

    return condition


def _transform_where_to_python(where_clause: str) -> str:
    """
    Tokenize WHERE clause and transform:
      - bare identifiers -> get('identifier')
      - '=' -> '=='
      - preserve and/or/not/in keywords (case-insensitive for logical ops)
    """
    token_re = re.compile(
        r"""
        (?P<WS>\s+)
        |(?P<STRING>'[^'\\]*(?:\\.[^'\\]*)*'|"[^"\\]*(?:\\.[^"\\]*)*")
        |(?P<OP><=|>=|!=|==|=|<|>)
        |(?P<LP>\()
        |(?P<RP>\))
        |(?P<COMMA>,)
        |(?P<NUM>\d+(?:\.\d+)?)
        |(?P<NAME>[A-Za-z_][A-Za-z0-9_]*)
        |(?P<OTHER>.)
        """,
        re.VERBOSE | re.DOTALL,
    )

    parts: List[str] = []
    pos = 0
    while pos < len(where_clause):
        m = token_re.match(where_clause, pos)
        if not m:
            raise ValueError(f"Unexpected token at position {pos}")
        pos = m.end()
        kind = m.lastgroup
        text = m.group()

        if kind == 'WS':
            parts.append(text)
        elif kind == 'STRING':
            parts.append(text)
        elif kind == 'NUM':
            parts.append(text)
        elif kind == 'OP':
            if text == '=':
                parts.append('==')
            else:
                parts.append(text)
        elif kind in ('LP', 'RP', 'COMMA'):
            parts.append(text)
        elif kind == 'NAME':
            lower = text.lower()
            if lower in ('and', 'or', 'not', 'in'):
                parts.append(lower)
            elif text in ('True', 'False', 'None'):
                parts.append(text)
            else:
                parts.append(f"get('{text}')")
        elif kind == 'OTHER':
            raise ValueError(f"Invalid character in WHERE clause: {text!r}")
        else:
            raise ValueError(f"Unhandled token: {text!r}")

    return ''.join(parts)


def _parse_order_by(order_clause: str) -> List[Dict[str, Any]]:
    """
    Parse ORDER BY clause into a list of {field, desc} dicts.
    Supports comma-separated fields, each optionally followed by ASC or DESC.
    """
    segments = _split_csv(order_clause)
    order_fields: List[Dict[str, Any]] = []
    for seg in segments:
        tokens = seg.strip().split()
        if not tokens:
            continue
        field = tokens[0]
        direction = tokens[1].upper() if len(tokens) > 1 else 'ASC'
        if direction not in ('ASC', 'DESC'):
            raise ValueError(f"Invalid ORDER BY direction for field '{field}': {direction}")
        order_fields.append({'field': field, 'desc': direction == 'DESC'})
    if not order_fields:
        raise ValueError("ORDER BY clause is empty.")
    return order_fields


def _apply_order_by(records: List[Dict[str, Any]], order_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply multi-key ORDER BY using stable sorts from last to first key.
    """
    result = list(records)
    for spec in reversed(order_fields):
        field = spec['field']
        desc = spec['desc']
        try:
            result.sort(key=lambda r: (r.get(field) is None, r.get(field)), reverse=desc)
        except TypeError:
            # Fall back to string comparison if types are not directly comparable
            result.sort(key=lambda r: (r.get(field) is None, str(r.get(field))), reverse=desc)
    return result
