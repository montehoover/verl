import re
import operator
from functools import partial


class _QueryParseError(ValueError):
    pass


def _strip_trailing_semicolon(s: str) -> str:
    s = s.strip()
    if s.endswith(";"):
        return s[:-1].rstrip()
    return s


def _split_csv(text: str) -> list:
    # Split by commas not inside quotes
    items = []
    buf = []
    in_quote = None
    escape = False
    for ch in text:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\" and in_quote is not None:
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
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf or text.endswith(","):
        items.append("".join(buf).strip())
    return [i for i in items if i != ""]


def _unquote_string(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        q = token[0]
        inner = token[1:-1]
        # simple backslash unescaping
        result = []
        escape = False
        for ch in inner:
            if escape:
                if ch in (q, "\\"):
                    result.append(ch)
                elif ch == "n":
                    result.append("\n")
                elif ch == "t":
                    result.append("\t")
                elif ch == "r":
                    result.append("\r")
                else:
                    # keep unknown escapes literally
                    result.append(ch)
                escape = False
            elif ch == "\\":
                escape = True
            else:
                result.append(ch)
        if escape:
            # trailing backslash, keep it
            result.append("\\")
        return "".join(result)
    return token


_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _parse_value(token: str):
    token = token.strip()
    # string literal
    if len(token) >= 2 and token[0] == token[-1] and token[0] in ("'", '"'):
        return _unquote_string(token)
    # booleans
    tl = token.lower()
    if tl == "true":
        return True
    if tl == "false":
        return False
    if tl in ("null", "none"):
        return None
    # numeric
    if _NUMERIC_RE.match(token):
        if re.match(r"^[+-]?\d+$", token):
            try:
                return int(token, 10)
            except Exception:
                return float(token)
        try:
            return float(token)
        except Exception:
            pass
    # bareword string
    if " " in token or "\t" in token:
        raise _QueryParseError("Invalid unquoted literal with spaces in WHERE clause")
    return token


_COND_RE = re.compile(
    r"^\s*([A-Za-z_][\w\.]*)\s*(=|!=|<>|<=|>=|<|>)\s*(.+?)\s*$", re.IGNORECASE
)


def _split_where_segments(where_str: str):
    # Split on AND/OR outside quotes, no parentheses support
    s = where_str.strip()
    if not s:
        raise _QueryParseError("Empty WHERE clause")
    segments = []
    ops = []
    in_quote = None
    escape = False
    i = 0
    seg_start = 0

    def is_word_boundary(idx):
        if idx <= 0:
            left_ok = True
        else:
            left_ok = not s[idx - 1].isalnum() and s[idx - 1] != "_"
        if idx >= len(s):
            right_ok = True
        else:
            right_ok = not s[idx].isalnum() and s[idx] != "_"
        return left_ok and right_ok

    while i < len(s):
        ch = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if in_quote:
            if ch == "\\":
                escape = True
            elif ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = ch
            i += 1
            continue

        # check for AND / OR
        if s[i:i + 3].upper() == "AND" and is_word_boundary(i) and is_word_boundary(i + 3):
            segments.append(s[seg_start:i].strip())
            ops.append("AND")
            i += 3
            seg_start = i
            continue
        if s[i:i + 2].upper() == "OR" and is_word_boundary(i) and is_word_boundary(i + 2):
            segments.append(s[seg_start:i].strip())
            ops.append("OR")
            i += 2
            seg_start = i
            continue
        i += 1

    last = s[seg_start:].strip()
    if last:
        segments.append(last)
    if not segments:
        raise _QueryParseError("Invalid WHERE clause")
    if len(segments) != len(ops) + 1:
        raise _QueryParseError("Malformed WHERE logical expression")
    return segments, ops


def _build_condition(segment: str):
    m = _COND_RE.match(segment)
    if not m:
        raise _QueryParseError(f"Invalid condition: {segment}")
    field = m.group(1)
    op = m.group(2)
    rhs_raw = m.group(3)
    rhs_val = _parse_value(rhs_raw)

    op_map = {
        "=": operator.eq,
        "!=": operator.ne,
        "<>": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }
    op_fn = op_map.get(op)
    if not op_fn:
        raise _QueryParseError(f"Unsupported operator: {op}")

    def comparator(lhs_val, rhs_val_inner, op_symbol):
        # for ordering comparisons, ensure comparable types
        ordering_ops = {"<", "<=", ">", ">="}
        eq_ops = {"=", "!=", "<>"}

        # Handle None
        if lhs_val is None or rhs_val_inner is None:
            if op_symbol in ordering_ops:
                # SQL treats comparisons with NULL as unknown -> here we raise
                raise _QueryParseError("Cannot perform ordered comparison with NULL")
            # For equality, None equals None only
            if op_symbol in ("=",):
                return lhs_val is None and rhs_val_inner is None
            else:
                return (lhs_val is None) != (rhs_val_inner is None)

        # If rhs is number and lhs is numeric string -> coerce
        if isinstance(rhs_val_inner, (int, float)) and isinstance(lhs_val, str):
            if _NUMERIC_RE.match(lhs_val.strip()):
                try:
                    lhs_val = float(lhs_val) if "." in lhs_val or "e" in lhs_val.lower() else int(lhs_val)
                except Exception:
                    pass

        # If both numeric
        if isinstance(lhs_val, (int, float)) and isinstance(rhs_val_inner, (int, float)):
            return op_map[op_symbol](lhs_val, rhs_val_inner)

        # If both strings
        if isinstance(lhs_val, str) and isinstance(rhs_val_inner, str):
            # Case-sensitive by default
            return op_map[op_symbol](lhs_val, rhs_val_inner)

        # If both booleans
        if isinstance(lhs_val, bool) and isinstance(rhs_val_inner, bool):
            return op_map[op_symbol](lhs_val, rhs_val_inner)

        # Mixed types
        if op_symbol in ordering_ops:
            raise _QueryParseError(f"Incomparable types for ordered comparison: {type(lhs_val).__name__} {op_symbol} {type(rhs_val_inner).__name__}")

        # For equality, fall back to string representation
        return op_map[op_symbol](str(lhs_val), str(rhs_val_inner))

    def cond_fn(record: dict) -> bool:
        lhs = record.get(field, None)
        return comparator(lhs, rhs_val, op)

    return cond_fn


def _parse_where(where_clause: str):
    segments, ops = _split_where_segments(where_clause)
    conds = [_build_condition(seg) for seg in segments]
    return conds, ops


def _apply_where_conds(records: list, conds, ops) -> list:
    def rec_matches(rec: dict) -> bool:
        result = conds[0](rec)
        for logic, cond in zip(ops, conds[1:]):
            if logic == "AND":
                if not result:
                    # short-circuit: False AND anything -> False
                    return False
                result = result and cond(rec)
            elif logic == "OR":
                if result:
                    # short-circuit: True OR anything -> True
                    return True
                result = cond(rec)
            else:
                raise _QueryParseError(f"Unknown logical operator {logic}")
        return result

    return [r for r in records if rec_matches(r)]


def _apply_where(records: list, where_clause: str) -> list:
    # Legacy helper kept for compatibility; not used by pipeline.
    conds, ops = _parse_where(where_clause)
    return _apply_where_conds(records, conds, ops)


def _normalize_sort_value(val):
    # Create sortable key with stable type ordering
    if val is None:
        return (4, 0)
    if isinstance(val, bool):
        return (2, 1 if val else 0)
    if isinstance(val, (int, float)):
        return (0, float(val))
    if isinstance(val, str):
        return (1, val)
    return (3, str(val))


def _parse_order_by(order_by_clause: str):
    parts = _split_csv(order_by_clause)
    if not parts:
        raise _QueryParseError("ORDER BY clause is empty")
    order_fields = []
    for part in parts:
        tokens = part.strip().split()
        if not tokens:
            continue
        field = tokens[0]
        direction = "ASC"
        if len(tokens) >= 2:
            dir_token = tokens[1].upper()
            if dir_token in ("ASC", "DESC"):
                direction = dir_token
            else:
                raise _QueryParseError(f"Invalid ORDER BY direction for '{field}': {tokens[1]}")
        order_fields.append((field, direction == "DESC"))
    return order_fields


def _apply_order_by_fields(records: list, order_fields) -> list:
    # Stable multi-key sort: sort by last key first
    result = list(records)
    for field, reverse in reversed(order_fields):
        result.sort(key=lambda r, f=field: _normalize_sort_value(r.get(f, None)), reverse=reverse)
    return result


def _apply_order_by(records: list, order_by_clause: str) -> list:
    # Legacy helper kept for compatibility; not used by pipeline.
    order_fields = _parse_order_by(order_by_clause)
    return _apply_order_by_fields(records, order_fields)


def _parse_select(select_clause: str):
    select_clause = select_clause.strip()
    if not select_clause:
        raise _QueryParseError("SELECT clause is empty")
    if select_clause == "*":
        return "*"
    fields = [f.strip() for f in _split_csv(select_clause)]
    if not all(fields):
        raise _QueryParseError("Invalid field list in SELECT")
    return fields


def _project_fields(records: list, fields):
    if fields == "*":
        # return shallow copies to avoid mutating inputs
        return [dict(r) for r in records]
    result = []
    for r in records:
        projected = {f: r.get(f, None) for f in fields}
        result.append(projected)
    return result


def parse_query_cmd(sql_query: str):
    """
    Pure function: parses the SQL-like query string and returns a plan dict.

    Returns:
        {
          "fields": list[str] or "*",
          "conds": list[callable] or None,
          "ops": list[str] or None,
          "order_fields": list[(field, desc: bool)] or None
        }
    Raises:
        _QueryParseError on invalid query
    """
    if not isinstance(sql_query, str) or not sql_query.strip():
        raise _QueryParseError("sql_query must be a non-empty string")

    query = _strip_trailing_semicolon(sql_query)

    pattern = re.compile(
        r"^\s*SELECT\s+(?P<select>.+?)"
        r"(?:\s+WHERE\s+(?P<where>.+?))?"
        r"(?:\s+ORDER\s+BY\s+(?P<order_by>.+))?\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.match(query)
    if not m:
        raise _QueryParseError("Invalid query format")

    select_clause = m.group("select")
    where_clause = m.group("where")
    order_by_clause = m.group("order_by")

    fields = _parse_select(select_clause)

    conds = ops = None
    if where_clause is not None:
        conds, ops = _parse_where(where_clause)

    order_fields = None
    if order_by_clause is not None:
        order_fields = _parse_order_by(order_by_clause)

    return {
        "fields": fields,
        "conds": conds,
        "ops": ops,
        "order_fields": order_fields,
    }


def _build_pipeline(plan: dict):
    """
    Build a list of pure transformation stages to apply to the dataset.
    Each stage is a function: records -> records
    """
    stages = []
    if plan.get("conds") is not None:
        stages.append(partial(_apply_where_conds, conds=plan["conds"], ops=plan["ops"]))
    if plan.get("order_fields") is not None:
        stages.append(partial(_apply_order_by_fields, order_fields=plan["order_fields"]))
    stages.append(partial(_project_fields, fields=plan["fields"]))
    return stages


def execute_parsed_query(dataset_list, plan: dict):
    """
    Pure function: executes a parsed query plan against the dataset.
    """
    if not isinstance(dataset_list, list) or any(not isinstance(x, dict) for x in dataset_list):
        raise _QueryParseError("dataset_list must be a list of dictionaries")

    records = list(dataset_list)
    for stage in _build_pipeline(plan):
        records = stage(records)
    return records


def execute_query_cmd(dataset_list, sql_query):
    """
    Execute a basic SQL-like statement over a list of dictionaries.

    Supported syntax (case-insensitive, FROM is omitted):
      SELECT field1, field2 [, ...]
      [WHERE field op value [AND|OR field op value] ...]
      [ORDER BY field1 [ASC|DESC] [, field2 [ASC|DESC] ...]]

    Examples:
      SELECT * WHERE age >= 30 ORDER BY last_name ASC, first_name DESC
      SELECT name, age WHERE city = 'Paris'
      SELECT id ORDER BY created_at DESC

    Arguments:
      - dataset_list: list[dict]
      - sql_query: str

    Returns: list[dict]
    Raises: ValueError on invalid query.
    """
    # Validate high-level input types and run parsed pipeline
    try:
        plan = parse_query_cmd(sql_query)
        result = execute_parsed_query(dataset_list, plan)
        return result
    except _QueryParseError as e:
        raise ValueError(str(e))
