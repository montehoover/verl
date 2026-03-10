from typing import List, Dict, Any, Optional, Callable, Mapping, Union
import re
import operator
from functools import partial


FilterType = Optional[Union[Callable[[Dict[str, Any]], bool], Mapping[str, Any]]]


def _matches_filters(record: Dict[str, Any], filters: FilterType) -> bool:
    if filters is None:
        return True
    if callable(filters):
        return bool(filters(record))
    if isinstance(filters, Mapping):
        for key, value in filters.items():
            if key not in record or record[key] != value:
                return False
        return True
    raise ValueError("filters must be None, a callable, or a mapping of field names to values")


def extract_fields(
    records: List[Dict[str, Any]],
    fields: List[str],
    filters: FilterType = None
) -> List[Dict[str, Any]]:
    """
    Extract specified fields from a list of record dictionaries, with optional filtering.

    Args:
        records: A list of dictionaries representing records.
        fields: A list of field names to extract from each record.
        filters: Optional filter conditions. Can be:
                 - None: no filtering
                 - Callable[[Dict[str, Any]], bool]: a predicate that returns True if the record should be included
                 - Mapping[str, Any]: equality-based filters; a record is included only if record[k] == v for all items

    Returns:
        A new list of dictionaries containing only the specified fields from records
        that satisfy the filter conditions.

    Raises:
        ValueError: If any item in records is not a dict, if a requested field is not
                    present in a matching record, or if filters is of an unsupported type.
    """
    result: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {idx} is not a dictionary")

        if not _matches_filters(record, filters):
            continue

        missing = [field for field in fields if field not in record]
        if missing:
            raise ValueError(
                f"Record at index {idx} is missing required field(s): {', '.join(missing)}"
            )

        result.append({field: record[field] for field in fields})

    return result


# -------------- Custom SQL-like query execution --------------

def execute_custom_query(data: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Execute a simple SQL-like query over a list of dictionaries.

    Supported syntax (case-insensitive keywords):
      SELECT <fields> [FROM <ignored>]
      [WHERE <condition>]
      [ORDER BY <field> [ASC|DESC] [, <field2> [ASC|DESC] ...]]

    - <fields> can be '*' or a comma-separated list of field names.
    - <condition> supports comparisons joined by AND/OR, e.g.:
        field = 10
        field != 'value'
        field >= 3.14
        field IN (1, 2, 3)
        field NOT IN ('a', 'b')
        Multiple conditions can be combined with AND/OR (left-to-right).
    - ORDER BY supports one or more fields with optional ASC/DESC.

    Args:
        data: List of record dictionaries.
        query: SQL-like query string.

    Returns:
        List of dictionaries as the result set.

    Raises:
        ValueError: On parse errors, unsupported constructs, missing fields, or type issues.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    select_part, where_part, order_part = _parse_query_sections(query)

    selected_fields = _parse_select_fields(select_part)

    # Build predicate from WHERE (if any)
    predicate: Callable[[Dict[str, Any]], bool]
    if where_part is None or where_part.strip() == "":
        predicate = lambda _rec: True
    else:
        predicate = _build_where_predicate(where_part)

    # Filter
    filtered: List[Dict[str, Any]] = []
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {idx} is not a dictionary")
        try:
            if predicate(record):
                filtered.append(record)
        except KeyError as e:
            raise ValueError(f"WHERE references missing field {e.args[0]!r} in record at index {idx}") from None
        except Exception as e:
            raise ValueError(f"Error evaluating WHERE for record at index {idx}: {e}") from None

    # Order
    ordered = list(filtered)
    if order_part:
        order_specs = _parse_order_by(order_part)
        # Validate fields exist on at least one record; we will error if a specific record is missing during sort key extraction.
        try:
            # Stable multi-pass sort from last key to first
            for field, reverse in reversed(order_specs):
                ordered.sort(key=_make_sort_key(field), reverse=reverse)
        except KeyError as e:
            raise ValueError(f"ORDER BY references missing field {e.args[0]!r}") from None
        except Exception as e:
            raise ValueError(f"Error during ORDER BY: {e}") from None

    # Project (SELECT)
    results: List[Dict[str, Any]] = []
    if selected_fields == ["*"]:
        results = [dict(rec) for rec in ordered]
    else:
        for idx, rec in enumerate(ordered):
            missing = [f for f in selected_fields if f not in rec]
            if missing:
                raise ValueError(
                    f"SELECT references missing field(s) {', '.join(missing)} in record at index {idx}"
                )
            results.append({f: rec[f] for f in selected_fields})

    return results


def _parse_query_sections(query: str) -> (str, Optional[str], Optional[str]):
    q = query.strip()
    uq = q.upper()

    if not uq.startswith("SELECT "):
        raise ValueError("Query must start with SELECT")

    select_start = len("SELECT ")
    # Find clause positions (first occurrence after SELECT)
    def find_kw(kw: str) -> int:
        return uq.find(f" {kw} ", select_start)

    idx_from = find_kw("FROM")
    idx_where = find_kw("WHERE")
    idx_order = uq.find(" ORDER BY ", select_start)

    # end of SELECT list is before the earliest of FROM/WHERE/ORDER BY (if present)
    candidates = [i for i in (idx_from, idx_where, idx_order) if i != -1]
    select_end = min(candidates) if candidates else len(q)
    select_part = q[select_start:select_end].strip()
    if not select_part:
        raise ValueError("SELECT field list is empty")

    # FROM is optional and ignored; compute its end if present
    if idx_from != -1:
        from_start = idx_from + len(" FROM ")
        # FROM ends before WHERE/ORDER or end
        after_from_candidates = [i for i in (idx_where, idx_order) if i != -1 and i > idx_from]
        from_end = min(after_from_candidates) if after_from_candidates else len(q)
        # from_identifier = q[from_start:from_end].strip()  # Ignored

    # WHERE
    where_part = None
    if idx_where != -1:
        where_start = idx_where + len(" WHERE ")
        where_end = idx_order if (idx_order != -1 and idx_order > idx_where) else len(q)
        where_part = q[where_start:where_end].strip()
        if not where_part:
            raise ValueError("WHERE clause is empty")

    # ORDER BY
    order_part = None
    if idx_order != -1:
        order_part = q[idx_order + len(" ORDER BY "):].strip()
        if not order_part:
            raise ValueError("ORDER BY clause is empty")

    return select_part, where_part, order_part


def _parse_select_fields(select_part: str) -> List[str]:
    if select_part == "*":
        return ["*"]
    fields = _split_by_commas_respecting_quotes(select_part)
    fields = [f.strip() for f in fields if f.strip()]
    if not fields:
        raise ValueError("No fields specified in SELECT")
    return fields


def _parse_order_by(order_part: str) -> List[tuple]:
    specs: List[tuple] = []
    for chunk in _split_by_commas_respecting_quotes(order_part):
        part = chunk.strip()
        if not part:
            continue
        pieces = part.split()
        if len(pieces) == 0:
            continue
        field = pieces[0]
        if not _is_identifier(field):
            raise ValueError(f"Invalid ORDER BY field: {field!r}")
        direction = pieces[1].upper() if len(pieces) > 1 else "ASC"
        if direction not in ("ASC", "DESC"):
            raise ValueError(f"Invalid ORDER BY direction for field {field!r}: {direction!r}")
        specs.append((field, direction == "DESC"))
    if not specs:
        raise ValueError("ORDER BY clause must specify at least one field")
    return specs


def _is_identifier(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_]\w*", s))


def _split_by_commas_respecting_quotes(s: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    in_quote: Optional[str] = None
    escape = False
    for ch in s:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch in ("'", '"'):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            buf.append(ch)
            continue
        if ch == "," and in_quote is None:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if in_quote is not None:
        raise ValueError("Unterminated string literal in list")
    if buf:
        parts.append("".join(buf))
    return parts


def _parse_literal(token: str) -> Any:
    t = token.strip()
    # Strings
    if (len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"')):
        # Unescape simple backslash escapes for the same quote type
        quote = t[0]
        inner = t[1:-1]
        inner = inner.replace(f"\\{quote}", quote).replace("\\\\", "\\")
        return inner
    # Booleans
    tl = t.lower()
    if tl == "true":
        return True
    if tl == "false":
        return False
    # Null
    if tl in ("null", "none"):
        return None
    # Numbers (int or float)
    if re.fullmatch(r"-?\d+", t):
        try:
            return int(t)
        except Exception:
            pass
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", t):
        try:
            return float(t)
        except Exception:
            pass
    # Fallback: raw token
    return t


def _build_where_predicate(where_part: str) -> Callable[[Dict[str, Any]], bool]:
    exprs, ops = _split_where_expressions(where_part)
    if not exprs:
        raise ValueError("Invalid WHERE clause")

    predicates: List[Callable[[Dict[str, Any]], bool]] = [ _parse_condition_expression(e) for e in exprs ]

    def predicate(record: Dict[str, Any]) -> bool:
        result = predicates[0](record)
        for op, fn in zip(ops, predicates[1:]):
            if op == "AND":
                result = result and fn(record)
            elif op == "OR":
                result = result or fn(record)
            else:
                raise ValueError(f"Unsupported logical operator: {op!r}")
        return result

    return predicate


def _split_where_expressions(s: str) -> (List[str], List[str]):
    """
    Split a WHERE clause into expressions and logical operators (AND/OR),
    respecting quotes and parentheses (e.g., for IN lists).
    """
    exprs: List[str] = []
    ops: List[str] = []

    buf: List[str] = []
    in_quote: Optional[str] = None
    escape = False
    paren_depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if escape:
            buf.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\":
            escape = True
            i += 1
            continue
        if ch in ("'", '"'):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            buf.append(ch)
            i += 1
            continue
        if ch == "(" and in_quote is None:
            paren_depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")" and in_quote is None and paren_depth > 0:
            paren_depth -= 1
            buf.append(ch)
            i += 1
            continue

        if in_quote is None and paren_depth == 0:
            # Check for AND/OR tokens with word boundaries
            if s[i:i+3].upper() == "AND" and _is_word_boundary(s, i-1) and _is_word_boundary(s, i+3):
                exprs.append("".join(buf).strip())
                buf = []
                ops.append("AND")
                i += 3
                continue
            if s[i:i+2].upper() == "OR" and _is_word_boundary(s, i-1) and _is_word_boundary(s, i+2):
                exprs.append("".join(buf).strip())
                buf = []
                ops.append("OR")
                i += 2
                continue

        buf.append(ch)
        i += 1

    if in_quote is not None:
        raise ValueError("Unterminated string literal in WHERE clause")
    if paren_depth != 0:
        raise ValueError("Unbalanced parentheses in WHERE clause")

    tail = "".join(buf).strip()
    if tail:
        exprs.append(tail)

    # Clean empty expressions (if any)
    exprs = [e for e in exprs if e]
    if len(exprs) == 0:
        raise ValueError("WHERE clause is empty or invalid")

    return exprs, ops


def _is_word_boundary(s: str, idx: int) -> bool:
    if idx < 0 or idx >= len(s):
        return True
    return not s[idx].isalnum() and s[idx] != "_"


def _parse_condition_expression(expr: str) -> Callable[[Dict[str, Any]], bool]:
    """
    Parse a single condition like:
      field >= 10
      field = 'x'
      field IN (1, 2, 3)
      field NOT IN ('a','b')
    """
    # IN / NOT IN
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s+(NOT\s+)?IN\s*\((.*)\)\s*", expr, flags=re.IGNORECASE | re.DOTALL)
    if m:
        field = m.group(1)
        not_token = m.group(2)
        items_raw = m.group(3)
        items = [_parse_literal(tok) for tok in _split_by_commas_respecting_quotes(items_raw)]
        container = items
        contains = partial(operator.contains, container)
        if not_token and not_token.strip().upper() == "NOT":
            return lambda rec: not contains(_get_field(rec, field))
        return lambda rec: contains(_get_field(rec, field))

    # Comparison operators
    m = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*(>=|<=|!=|==|=|>|<)\s*(.+?)\s*", expr, flags=re.DOTALL)
    if m:
        field, op_str, rhs = m.group(1), m.group(2), m.group(3)
        value = _parse_literal(rhs)
        ops_map = {
            "=": operator.eq,
            "==": operator.eq,
            "!=": operator.ne,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
        }
        if op_str not in ops_map:
            raise ValueError(f"Unsupported operator: {op_str!r}")
        cmp_op = ops_map[op_str]

        def predicate(rec: Dict[str, Any]) -> bool:
            left = _get_field(rec, field)
            try:
                return bool(cmp_op(left, value))
            except Exception as e:
                raise ValueError(f"Cannot compare values for field {field!r}: {e}")

        return predicate

    raise ValueError(f"Invalid condition in WHERE: {expr!r}")


def _get_field(rec: Dict[str, Any], field: str) -> Any:
    if field not in rec:
        raise KeyError(field)
    return rec[field]


def _make_sort_key(field: str) -> Callable[[Dict[str, Any]], Any]:
    def key_fn(rec: Dict[str, Any]) -> Any:
        if field not in rec:
            raise KeyError(field)
        return rec[field]
    return key_fn
