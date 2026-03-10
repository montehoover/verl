import re
import operator
from functools import partial
from typing import Any, Iterable, Mapping, Sequence, List, Dict, Callable, Tuple
from collections.abc import Sequence as _Sequence

__all__ = ["extract_fields", "filter_and_extract", "run_custom_query"]


def extract_fields(records: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Return a new list of dictionaries containing only the specified fields from each input record.

    - If a requested field is missing from a record, it is omitted from that record's result.
    - Duplicate field names in `fields` are ignored (first occurrence order is preserved).

    Args:
        records: An iterable of dictionaries (mappings) representing the dataset.
        fields: A sequence of field names to extract.

    Returns:
        A list of dictionaries containing only the requested fields for each input record.

    Raises:
        TypeError: If any record is not a mapping or fields is not a sequence of strings.

    Example:
        >>> data = [
        ...     {"id": 1, "name": "Alice", "age": 30},
        ...     {"id": 2, "name": "Bob", "city": "NYC"},
        ... ]
        >>> extract_fields(data, ["id", "name"])
        [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
    """
    # Validate fields are strings and deduplicate while preserving order
    unique_fields: List[str] = []
    for f in fields:
        if not isinstance(f, str):
            raise TypeError("All field names must be strings.")
        if f not in unique_fields:
            unique_fields.append(f)

    result: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            raise TypeError("Each record must be a mapping (e.g., dict).")
        projected = {k: rec[k] for k in unique_fields if k in rec}
        result.append(projected)

    return result


def filter_and_extract(
    records: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    conditions: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """
    Filter records by conditions and then project only the requested fields.

    The `conditions` mapping specifies per-field constraints. Each value in `conditions`
    can be one of the following forms:
      - Literal: record[field] must equal the literal value.
      - Callable: a predicate that receives record[field] and returns True/False.
      - Sequence (list/tuple/set) of allowed values: record[field] must be in the sequence.
      - Tuple(operator, operand): supported operators are
            "==", "!=", ">", ">=", "<", "<=",
            "in", "not in", "contains", "icontains",
            "startswith", "endswith"
        Examples:
            ("in", {1, 2, 3}), (">", 10), ("icontains", "abc")
      - Operator dict: a dict of operators to operands (all must match), using keys:
            "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",
            "$in", "$nin", "$contains", "$icontains",
            "$startswith", "$endswith"

    Notes:
      - If a condition references a field that is missing in a record, that record does not match.

    Args:
        records: Iterable of mapping records to filter and project.
        fields: Sequence of field names to include in the output.
        conditions: Mapping of field -> condition (see forms above).

    Returns:
        List of projected dictionaries for records that satisfy all conditions.

    Raises:
        TypeError: If inputs are of incorrect types or unsupported operators are used.

    Example:
        >>> data = [
        ...     {"id": 1, "name": "Alice", "age": 30, "city": "NYC"},
        ...     {"id": 2, "name": "Bob", "age": 25, "city": "LA"},
        ...     {"id": 3, "name": "Carol", "age": 35, "city": "NYC"},
        ... ]
        >>> filter_and_extract(data, ["id", "name"], {"city": "NYC", "age": (">=", 30)})
        [{'id': 1, 'name': 'Alice'}, {'id': 3, 'name': 'Carol'}]
    """
    if not isinstance(conditions, Mapping):
        raise TypeError("conditions must be a mapping of field -> condition")

    # Validate fields are strings and deduplicate while preserving order
    unique_fields: List[str] = []
    for f in fields:
        if not isinstance(f, str):
            raise TypeError("All field names must be strings.")
        if f not in unique_fields:
            unique_fields.append(f)

    def _is_sequence_but_not_str(x: Any) -> bool:
        return isinstance(x, _Sequence) and not isinstance(x, (str, bytes, bytearray))

    def _apply_operator(op: str, value: Any, operand: Any) -> bool:
        if op == "==":
            return value == operand
        if op == "!=":
            return value != operand
        if op == ">":
            return value > operand  # type: ignore[operator]
        if op == ">=":
            return value >= operand  # type: ignore[operator]
        if op == "<":
            return value < operand  # type: ignore[operator]
        if op == "<=":
            return value <= operand  # type: ignore[operator]
        if op == "in":
            return value in operand  # operand should be a container
        if op == "not in":
            return value not in operand
        if op == "contains":
            # record value must contain operand
            try:
                return operand in value
            except TypeError:
                return False
        if op == "icontains":
            # case-insensitive contains; stringify both sides
            try:
                return str(operand).lower() in str(value).lower()
            except Exception:
                return False
        if op == "startswith":
            try:
                return str(value).startswith(str(operand))
            except Exception:
                return False
        if op == "endswith":
            try:
                return str(value).endswith(str(operand))
            except Exception:
                return False
        raise TypeError(f"Unsupported operator: {op}")

    def _evaluate_condition(v: Any, cond: Any) -> bool:
        # Callable predicate
        if callable(cond):
            return bool(cond(v))

        # Operator dict
        if isinstance(cond, Mapping):
            for k, operand in cond.items():
                if k == "$eq":
                    if not _apply_operator("==", v, operand):
                        return False
                elif k == "$ne":
                    if not _apply_operator("!=", v, operand):
                        return False
                elif k == "$gt":
                    if not _apply_operator(">", v, operand):
                        return False
                elif k == "$gte":
                    if not _apply_operator(">=", v, operand):
                        return False
                elif k == "$lt":
                    if not _apply_operator("<", v, operand):
                        return False
                elif k == "$lte":
                    if not _apply_operator("<=", v, operand):
                        return False
                elif k == "$in":
                    if not _apply_operator("in", v, operand):
                        return False
                elif k == "$nin":
                    if not _apply_operator("not in", v, operand):
                        return False
                elif k == "$contains":
                    if not _apply_operator("contains", v, operand):
                        return False
                elif k == "$icontains":
                    if not _apply_operator("icontains", v, operand):
                        return False
                elif k == "$startswith":
                    if not _apply_operator("startswith", v, operand):
                        return False
                elif k == "$endswith":
                    if not _apply_operator("endswith", v, operand):
                        return False
                else:
                    raise TypeError(f"Unsupported operator key in condition dict: {k}")
            return True

        # Tuple operator
        if isinstance(cond, tuple) and len(cond) == 2 and isinstance(cond[0], str):
            op, operand = cond  # type: Tuple[str, Any]
            return _apply_operator(op, v, operand)

        # Sequence of allowed values (membership)
        if _is_sequence_but_not_str(cond):
            try:
                return v in cond  # type: ignore[operator]
            except Exception:
                return False

        # Literal equality
        return v == cond

    result: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            raise TypeError("Each record must be a mapping (e.g., dict).")

        # Check conditions
        matched = True
        for field, cond in conditions.items():
            if field not in rec:
                matched = False
                break
            if not _evaluate_condition(rec[field], cond):
                matched = False
                break

        if not matched:
            continue

        # Project fields
        projected = {k: rec[k] for k in unique_fields if k in rec}
        result.append(projected)

    return result


# -------------------------
# SQL-like query processing
# -------------------------

_MISSING = object()

def _get_field_value(rec: Mapping[str, Any], path: str) -> Any:
    """Retrieve a possibly dotted field from a record. Returns _MISSING if not found."""
    if not path or not isinstance(path, str):
        return _MISSING
    cur: Any = rec
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _is_identifier(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*$", name))


def _unescape_string_literal(s: str) -> str:
    # Handle simple backslash escapes for the matching quote char and backslash
    return s.replace(r"\\", "\\").replace(r"\'", "'").replace(r'\"', '"')


def _parse_value(token: str) -> Any:
    t = token.strip()
    if not t:
        raise ValueError("Empty value in query.")
    if (t[0] == t[-1]) and t[0] in ("'", '"') and len(t) >= 2:
        return _unescape_string_literal(t[1:-1])
    # Booleans
    if t.lower() == "true":
        return True
    if t.lower() == "false":
        return False
    # Nulls
    if t.lower() in ("null", "none"):
        return None
    # Numbers
    if re.fullmatch(r"-?\d+", t):
        try:
            return int(t)
        except Exception:
            pass
    if re.fullmatch(r"-?\d+\.\d*", t) or re.fullmatch(r"-?\d*\.\d+", t):
        try:
            return float(t)
        except Exception:
            pass
    # Bare identifier treated as a string literal
    return t


def _split_outside_quotes_and_parens(s: str, sep: str) -> List[str]:
    """Split string by a single-character separator outside quotes and parentheses."""
    assert len(sep) == 1
    parts: List[str] = []
    buf: List[str] = []
    quote: str | None = None
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if quote:
            if ch == "\\" and i + 1 < len(s):
                buf.append(ch)
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
                buf.append(ch)
                i += 1
                continue
            buf.append(ch)
            i += 1
            continue
        else:
            if ch in ("'", '"'):
                quote = ch
                buf.append(ch)
                i += 1
                continue
            if ch == "(":
                depth += 1
                buf.append(ch)
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                buf.append(ch)
                i += 1
                continue
            if ch == sep and depth == 0:
                parts.append("".join(buf).strip())
                buf = []
                i += 1
                continue
            buf.append(ch)
            i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p != ""]


def _split_top_level_and(s: str) -> List[str]:
    """Split a WHERE clause on AND at top level (outside quotes/parentheses)."""
    parts: List[str] = []
    buf: List[str] = []
    quote: str | None = None
    depth = 0
    i = 0
    n = len(s)

    def is_word_boundary(idx: int) -> bool:
        if idx < 0 or idx >= n:
            return True
        return not s[idx].isalnum() and s[idx] != "_"

    while i < n:
        ch = s[i]
        if quote:
            if ch == "\\" and i + 1 < n:
                buf.append(ch)
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
                buf.append(ch)
                i += 1
                continue
            buf.append(ch)
            i += 1
            continue
        else:
            if ch in ("'", '"'):
                quote = ch
                buf.append(ch)
                i += 1
                continue
            if ch == "(":
                depth += 1
                buf.append(ch)
                i += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
                buf.append(ch)
                i += 1
                continue
            # Check for AND at top level
            if depth == 0 and (i + 3) <= n and s[i:i+3].upper() == "AND" and is_word_boundary(i - 1) and is_word_boundary(i + 3):
                parts.append("".join(buf).strip())
                buf = []
                i += 3
                continue
            buf.append(ch)
            i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p != ""]


def _parse_value_list(token: str) -> List[Any]:
    t = token.strip()
    if not (t.startswith("(") and t.endswith(")")):
        raise ValueError("IN list must be enclosed in parentheses.")
    inner = t[1:-1]
    items = _split_outside_quotes_and_parens(inner, ",")
    return [_parse_value(it) for it in items]


def _parse_order_by(order_clause: str) -> List[Tuple[str, bool]]:
    """Parse ORDER BY clause into a list of (field, asc) tuples."""
    items = _split_outside_quotes_and_parens(order_clause, ",")
    result: List[Tuple[str, bool]] = []
    for item in items:
        m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(ASC|DESC)?\s*$", item, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Malformed ORDER BY item: {item!r}")
        field = m.group(1)
        if not _is_identifier(field):
            raise ValueError(f"Invalid field in ORDER BY: {field!r}")
        direction = m.group(2).upper() if m.group(2) else "ASC"
        asc = direction == "ASC"
        result.append((field, asc))
    return result


def _compile_condition(cond_str: str) -> Tuple[str, Callable[[Mapping[str, Any]], bool]]:
    """
    Compile a single condition like "age >= 21" into a predicate function.
    Returns (field, predicate).
    """
    m = re.match(
        r"""^\s*
            (?P<field>[A-Za-z_][A-Za-z0-9_\.]*)
            \s*
            (?P<op>!=|==|=|>=|<=|>|<|IN|NOT\s+IN|CONTAINS|ICONTAINS|STARTSWITH|ENDSWITH)
            \s*
            (?P<rhs>.+?)
            \s*$
        """,
        cond_str,
        flags=re.IGNORECASE | re.VERBOSE,
    )
    if not m:
        raise ValueError(f"Malformed condition: {cond_str!r}")

    field = m.group("field")
    if not _is_identifier(field):
        raise ValueError(f"Invalid field name in condition: {field!r}")

    op_raw = m.group("op").upper().replace(" ", "")
    rhs_raw = m.group("rhs").strip()

    if op_raw in ("=", "=="):
        op = "=="
        rhs = _parse_value(rhs_raw)
        def pred(rec: Mapping[str, Any]) -> bool:
            v = _get_field_value(rec, field)
            if v is _MISSING:
                return False
            try:
                return operator.eq(v, rhs)
            except Exception:
                return False
    elif op_raw == "!=":
        rhs = _parse_value(rhs_raw)
        def pred(rec: Mapping[str, Any]) -> bool:
            v = _get_field_value(rec, field)
            if v is _MISSING:
                return False
            try:
                return operator.ne(v, rhs)
            except Exception:
                return False
    elif op_raw in (">", ">=", "<", "<="):
        rhs = _parse_value(rhs_raw)
        op_map = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
        cmp_op = op_map[op_raw]
        def pred(rec: Mapping[str, Any]) -> bool:
            v = _get_field_value(rec, field)
            if v is _MISSING:
                return False
            try:
                return bool(cmp_op(v, rhs))
            except Exception:
                return False
    elif op_raw in ("IN", "NOTIN"):
        rhs_list = _parse_value_list(rhs_raw)
        is_not = (op_raw == "NOTIN")
        def pred(rec: Mapping[str, Any]) -> bool:
            v = _get_field_value(rec, field)
            if v is _MISSING:
                return False
            try:
                ok = v in rhs_list
            except Exception:
                ok = False
            return (not ok) if is_not else ok
    elif op_raw == "CONTAINS":
        rhs = _parse_value(rhs_raw)
        def pred(rec: Mapping[str, Any]) -> bool:
            v = _get_field_value(rec, field)
            if v is _MISSING:
                return False
            try:
                return rhs in v
            except Exception:
                return False
    elif op_raw == "ICONTAINS":
        rhs = str(_parse_value(rhs_raw)).lower()
        def pred(rec: Mapping[str, Any]) -> bool:
            v = _get_field_value(rec, field)
            if v is _MISSING:
                return False
            try:
                return rhs in str(v).lower()
            except Exception:
                return False
    elif op_raw == "STARTSWITH":
        rhs = str(_parse_value(rhs_raw))
        def pred(rec: Mapping[str, Any]) -> bool:
            v = _get_field_value(rec, field)
            if v is _MISSING:
                return False
            try:
                return str(v).startswith(rhs)
            except Exception:
                return False
    elif op_raw == "ENDSWITH":
        rhs = str(_parse_value(rhs_raw))
        def pred(rec: Mapping[str, Any]) -> bool:
            v = _get_field_value(rec, field)
            if v is _MISSING:
                return False
            try:
                return str(v).endswith(rhs)
            except Exception:
                return False
    else:
        raise ValueError(f"Unsupported operator in condition: {m.group('op')!r}")

    return field, pred


def run_custom_query(dataset: Iterable[Mapping[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Execute a simple SQL-like query over a list of dictionaries.

    Supported syntax (case-insensitive keywords):
      SELECT field1, field2, ...            -- Use * for all fields
      [WHERE <cond1> AND <cond2> AND ...]   -- Conditions combined with AND
      [ORDER BY field [ASC|DESC], ...]

    Supported condition operators:
      =, !=, >, >=, <, <=, IN (...), NOT IN (...),
      CONTAINS 'substr', ICONTAINS 'substr', STARTSWITH 'p', ENDSWITH 's'

    Returns:
      List of dictionaries (projected per SELECT). Raises ValueError for malformed queries
      or processing errors.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    # Strip optional trailing semicolon
    q = query.strip().rstrip(";").strip()

    try:
        m = re.match(
            r"""^\s*SELECT\s+
                (?P<select>.+?)
                (?:\s+WHERE\s+(?P<where>.+?))?
                (?:\s+ORDER\s+BY\s+(?P<order>.+?))?
                \s*$""",
            q,
            flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        if not m:
            raise ValueError("Malformed query. Expected SELECT ... [WHERE ...] [ORDER BY ...]")

        select_clause = m.group("select").strip()
        where_clause = m.group("where")
        order_clause = m.group("order")

        # Parse SELECT
        select_all = False
        select_fields: List[str] = []
        if select_clause == "*":
            select_all = True
        else:
            select_fields = _split_outside_quotes_and_parens(select_clause, ",")
            if not select_fields:
                raise ValueError("SELECT clause must specify at least one field or *")
            for f in select_fields:
                if not _is_identifier(f):
                    raise ValueError(f"Invalid field in SELECT: {f!r}")

        # Compile WHERE predicate
        predicates: List[Callable[[Mapping[str, Any]], bool]] = []
        if where_clause and where_clause.strip():
            cond_strs = _split_top_level_and(where_clause.strip())
            if not cond_strs:
                raise ValueError("Malformed WHERE clause.")
            for cond in cond_strs:
                _, pred = _compile_condition(cond)
                predicates.append(pred)

        def record_matches(rec: Mapping[str, Any]) -> bool:
            for p in predicates:
                if not p(rec):
                    return False
            return True

        # Parse ORDER BY specs
        order_specs: List[Tuple[str, bool]] = []
        if order_clause and order_clause.strip():
            order_specs = _parse_order_by(order_clause.strip())

        # Ensure dataset items are mappings and filter
        filtered_records: List[Mapping[str, Any]] = []
        for rec in dataset:
            if not isinstance(rec, Mapping):
                raise ValueError("All dataset items must be mappings (e.g., dict).")
            if record_matches(rec):
                filtered_records.append(rec)

        # Apply ORDER BY (stable multi-key sort, from last to first)
        if order_specs:
            try:
                for field, asc in reversed(order_specs):
                    def keyfn(r: Mapping[str, Any], _field=field):
                        v = _get_field_value(r, _field)
                        return (1, None) if v is _MISSING else (0, v)
                    filtered_records.sort(key=keyfn, reverse=not asc)
            except TypeError as te:
                raise ValueError(f"ORDER BY comparison failed: {te}") from te
            except Exception as ex:
                raise ValueError(f"ORDER BY failed: {ex}") from ex

        # Project SELECT
        results: List[Dict[str, Any]] = []
        if select_all:
            for rec in filtered_records:
                results.append(dict(rec))
        else:
            for rec in filtered_records:
                projected: Dict[str, Any] = {}
                for f in select_fields:
                    v = _get_field_value(rec, f)
                    if v is not _MISSING:
                        projected[f] = v
                results.append(projected)

        return results

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to process query: {e}") from e
