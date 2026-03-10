import re
import operator
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Tuple


def run_sql_query(records: List[Dict[str, Any]], command: str) -> List[Dict[str, Any]]:
    """
    Execute a basic SQL-like statement against a list of dict records.
    Supported clauses:
      - SELECT <col1, col2, ... | *>
      - optional FROM <ignored_identifier>
      - optional WHERE <conditions with AND/OR, =, !=, <, <=, >, >=, LIKE, IN, NOT LIKE, NOT IN, IS NULL, IS NOT NULL>
      - optional ORDER BY <col [ASC|DESC], ...>

    :param records: list of dictionaries
    :param command: SQL-like string
    :return: list of dictionaries with selected columns
    :raises ValueError: on invalid or unprocessable command
    """
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("records must be a list of dictionaries")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")

    cmd = command.strip()

    # Parse the command
    pattern = re.compile(
        r"""^\s*SELECT\s+
            (?P<select>.+?)(?=\s+FROM\s+|\s+WHERE\s+|\s+ORDER\s+BY\s+|$)
            (?:\s+FROM\s+(?P<from>\w+))?
            (?:\s+WHERE\s+(?P<where>.+?)(?=\s+ORDER\s+BY\s+|$))?
            (?:\s+ORDER\s+BY\s+(?P<order>.+))?
            \s*;?\s*$""",
        re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )
    m = pattern.match(cmd)
    if not m:
        raise ValueError("Invalid query format")

    select_clause = m.group("select").strip()
    where_clause = (m.group("where") or "").strip()
    order_clause = (m.group("order") or "").strip()

    # Parse SELECT
    if not select_clause:
        raise ValueError("SELECT clause is required")
    select_all = select_clause == "*"
    if not select_all:
        select_fields = [c.strip() for c in select_clause.split(",") if c.strip()]
        if not select_fields:
            raise ValueError("No columns specified in SELECT clause")
    else:
        select_fields = []

    # Parse WHERE into a predicate
    predicate = _parse_where(where_clause) if where_clause else (lambda r: True)

    # Filter
    try:
        filtered = [r for r in records if predicate(r)]
    except Exception as e:
        raise ValueError(f"Failed to evaluate WHERE clause: {e}") from e

    # Order
    ordered = _apply_order_by(filtered, order_clause) if order_clause else filtered

    # Project (SELECT)
    result: List[Dict[str, Any]] = []
    if select_all:
        # Return shallow copies to avoid accidental mutation from callers
        result = [dict(row) for row in ordered]
    else:
        for row in ordered:
            projected = {col: row.get(col, None) for col in select_fields}
            result.append(projected)

    return result


# ---------- Helpers ----------

def _parse_where(where: str) -> Callable[[Dict[str, Any]], bool]:
    """
    Parse a WHERE clause string into a predicate function.
    Supports:
      - Binary ops: =, ==, !=, <>, <, <=, >, >=
      - LIKE, NOT LIKE (SQL wildcards: % and _)
      - IN (...), NOT IN (...)
      - IS NULL, IS NOT NULL
      - Boolean connectors: AND, OR (AND has higher precedence)
    """
    tokens = _tokenize_where(where)
    if not tokens:
        raise ValueError("Empty WHERE clause")

    # Convert tokens to postfix (RPN) using shunting-yard (AND > OR)
    output: List[Any] = []
    ops_stack: List[str] = []
    precedence = {"OR": 1, "AND": 2}

    for tk in tokens:
        if isinstance(tk, str) and tk in ("AND", "OR"):
            while ops_stack and precedence.get(ops_stack[-1], 0) >= precedence[tk]:
                output.append(ops_stack.pop())
            ops_stack.append(tk)
        else:
            # tk is a condition string
            output.append(_compile_condition(tk))

    while ops_stack:
        output.append(ops_stack.pop())

    # Build predicate from RPN
    def combine(op: str, a: Callable, b: Callable) -> Callable:
        if op == "AND":
            return lambda rec: a(rec) and b(rec)
        else:
            return lambda rec: a(rec) or b(rec)

    stack: List[Callable[[Dict[str, Any]], bool]] = []
    for item in output:
        if callable(item):
            stack.append(item)
        else:
            # operator
            if len(stack) < 2:
                raise ValueError("Invalid WHERE logical expression")
            b = stack.pop()
            a = stack.pop()
            stack.append(combine(item, a, b))

    if len(stack) != 1:
        raise ValueError("Invalid WHERE expression")

    return stack[0]


def _tokenize_where(where: str) -> List[Any]:
    """
    Tokenize WHERE clause into a list containing either:
      - condition strings
      - 'AND' or 'OR'
    Avoid splitting inside quotes or parentheses.
    """
    s = where.strip()
    if not s:
        return []

    tokens: List[Any] = []
    buf: List[str] = []
    i = 0
    n = len(s)
    in_quote: str = ""
    paren_depth = 0

    def flush_buf():
        nonlocal buf
        if buf:
            tokens.append("".join(buf).strip())
            buf = []

    while i < n:
        ch = s[i]

        # Handle quotes (single or double)
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                # Check for escaped quote (two quotes in a row)
                if i + 1 < n and s[i + 1] == in_quote:
                    # It's an escaped quote - keep one and skip the next
                    i += 1
                    buf.append(s[i])
                else:
                    in_quote = ""
            i += 1
            continue

        if ch in ("'", '"'):
            in_quote = ch
            buf.append(ch)
            i += 1
            continue

        # Track parentheses for IN lists
        if ch == "(":
            paren_depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            buf.append(ch)
            i += 1
            continue

        # Detect AND/OR outside quotes and parentheses
        if paren_depth == 0 and ch.isspace():
            # Peek ahead to see if next word is AND/OR
            j = i
            while j < n and s[j].isspace():
                j += 1
            k = j
            while k < n and s[k].isalpha():
                k += 1
            word = s[j:k].upper()
            if word in ("AND", "OR"):
                flush_buf()
                tokens.append(word)
                i = k
                continue
        buf.append(ch)
        i += 1

    flush_buf()
    # Remove empty tokens
    tokens = [t for t in tokens if not isinstance(t, str) or t.strip()]
    # Validate sequence (cond (op cond)*)
    if not tokens:
        return []
    # Disallow leading or trailing AND/OR
    if isinstance(tokens[0], str) and tokens[0] in ("AND", "OR"):
        raise ValueError("WHERE clause cannot start with a logical operator")
    if isinstance(tokens[-1], str) and tokens[-1] in ("AND", "OR"):
        raise ValueError("WHERE clause cannot end with a logical operator")
    # Disallow consecutive logical operators
    for a, b in zip(tokens, tokens[1:]):
        if isinstance(a, str) and a in ("AND", "OR") and isinstance(b, str) and b in ("AND", "OR"):
            raise ValueError("Consecutive logical operators in WHERE clause")
    return tokens


def _compile_condition(cond: str) -> Callable[[Dict[str, Any]], bool]:
    """
    Compile a single condition string into a predicate function.
    """
    s = cond.strip()

    # Normalize multiple spaces to single around keywords like IS NOT, NOT IN, NOT LIKE
    s_norm = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()

    # Patterns ordered to match longest operators first
    # Field names allow letters, numbers, underscore, and dot (for nested-ish fields)
    field_pat = r"(?P<field>[A-Za-z_][A-Za-z0-9_\.]*)"
    ops = [
        r"IS\s+NOT",
        r"IS",
        r"NOT\s+LIKE",
        r"LIKE",
        r"NOT\s+IN",
        r"IN",
        r">=",
        r"<=",
        r"!=",
        r"<>",
        r"==",
        r"=",
        r">",
        r"<",
    ]
    op_group = "(?P<op>" + "|".join(ops) + ")"
    pattern = re.compile(rf"^{field_pat}\s*{op_group}\s*(?P<rhs>.+)$", re.IGNORECASE)

    m = pattern.match(s_norm)
    if m:
        field = m.group("field")
        op = m.group("op").upper()
        rhs_raw = m.group("rhs").strip()

        if op == "IS":
            if rhs_raw.upper() != "NULL":
                raise ValueError("IS operator only supports NULL")
            return lambda rec: _get_field(rec, field) is None

        if op == "IS NOT":
            if rhs_raw.upper() != "NULL":
                raise ValueError("IS NOT operator only supports NULL")
            return lambda rec: _get_field(rec, field) is not None

        if op in ("IN", "NOT IN"):
            values = _parse_in_list(rhs_raw)
            if values is None:
                raise ValueError("Invalid IN list syntax")
            values_set = set(values)
            if op == "IN":
                return lambda rec: _normalize_value(_get_field(rec, field)) in values_set
            else:
                return lambda rec: _normalize_value(_get_field(rec, field)) not in values_set

        if op in ("LIKE", "NOT LIKE"):
            pattern_re = _like_to_regex(rhs_raw)
            if op == "LIKE":
                return lambda rec: _match_like(pattern_re, _get_field(rec, field))
            else:
                return lambda rec: not _match_like(pattern_re, _get_field(rec, field))

        # Binary numeric/string comparisons
        op_map = {
            "=": operator.eq,
            "==": operator.eq,
            "!=": operator.ne,
            "<>": operator.ne,
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
        }
        if op not in op_map:
            raise ValueError(f"Unsupported operator: {op}")

        rhs_val = _parse_literal(rhs_raw)

        def pred(rec: Dict[str, Any]) -> bool:
            left = _get_field(rec, field)
            return _compare_values(left, rhs_val, op_map[op])

        return pred

    # If nothing matched, try shorthand boolean field e.g., "active" or "NOT active" (not documented but useful)
    m_bool = re.match(rf"^(?P<neg>NOT\s+)?{field_pat}\s*$", s_norm, re.IGNORECASE)
    if m_bool:
        field = m_bool.group("field")
        neg = bool(m_bool.group("neg"))
        if neg:
            return lambda rec: not bool(_get_field(rec, field))
        else:
            return lambda rec: bool(_get_field(rec, field))

    raise ValueError(f"Invalid condition: {cond}")


def _parse_literal(token: str) -> Any:
    t = token.strip()
    # Quoted string
    if (len(t) >= 2) and ((t[0] == t[-1] == "'") or (t[0] == t[-1] == '"')):
        # Unescape doubled quotes
        quote = t[0]
        inner = t[1:-1]
        inner = inner.replace(quote * 2, quote)
        return inner
    # NULL / NONE
    if t.upper() in ("NULL", "NONE"):
        return None
    # Booleans
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    # Numbers: int or float
    if re.fullmatch(r"[+-]?\d+", t):
        try:
            return int(t)
        except Exception:
            pass
    if re.fullmatch(r"[+-]?\d+\.\d*", t) or re.fullmatch(r"[+-]?\d*\.\d+", t):
        try:
            return float(t)
        except Exception:
            pass
    # Fallback: bareword string
    return t


def _parse_in_list(rhs: str) -> List[Any] | None:
    s = rhs.strip()
    if not (s.startswith("(") and s.endswith(")")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return []
    # Split by commas outside quotes/parentheses
    parts = _split_by_comma(inner)
    values = [_parse_literal(p) for p in parts]
    # Normalize for membership checks (coerce to consistent types)
    return [_normalize_value(v) for v in values]


def _split_by_comma(s: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    in_quote: str = ""
    i = 0
    n = len(s)
    paren = 0
    while i < n:
        ch = s[i]
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                if i + 1 < n and s[i + 1] == in_quote:
                    i += 1
                    buf.append(s[i])
                else:
                    in_quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            paren += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            paren = max(0, paren - 1)
            buf.append(ch)
            i += 1
            continue
        if ch == "," and paren == 0:
            out.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        out.append("".join(buf).strip())
    return out


def _get_field(rec: Dict[str, Any], field: str) -> Any:
    # Support dot-notation for nested dicts
    parts = field.split(".")
    val: Any = rec
    for p in parts:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            return None
    return val


def _normalize_value(v: Any) -> Any:
    # Normalize for set membership and LIKE matching
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    return str(v)


def _like_to_regex(rhs: str) -> re.Pattern:
    pat_raw = _parse_literal(rhs)
    if pat_raw is None:
        pat_raw = "NULL"
    s = str(pat_raw)
    # Escape regex meta, then replace SQL wildcards
    escaped = ""
    for ch in s:
        if ch == "%":
            escaped += ".*"
        elif ch == "_":
            escaped += "."
        else:
            escaped += re.escape(ch)
    regex = f"^{escaped}$"
    return re.compile(regex, re.IGNORECASE)


def _match_like(pattern_re: re.Pattern, val: Any) -> bool:
    if val is None:
        return False
    return bool(pattern_re.match(str(val)))


def _compare_values(left: Any, right: Any, op_func: Callable[[Any, Any], bool]) -> bool:
    # Attempt numeric comparison if both are numbers
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return op_func(left, right)
    # Try to coerce numeric-like strings
    if isinstance(left, str) and isinstance(right, (int, float)):
        try:
            lv = float(left) if "." in left or "e" in left.lower() else int(left)
            return op_func(lv, right)
        except Exception:
            pass
    if isinstance(right, str) and isinstance(left, (int, float)):
        try:
            rv = float(right) if "." in right or "e" in right.lower() else int(right)
            return op_func(left, rv)
        except Exception:
            pass
    # Fallback to string comparison
    lstr = "" if left is None else str(left)
    rstr = "" if right is None else str(right)
    # Case-insensitive for equality comparisons; order comparisons use case-sensitive string order
    if op_func in (operator.eq, operator.ne):
        return op_func(lstr.lower(), rstr.lower())
    else:
        return op_func(lstr, rstr)


def _apply_order_by(rows: List[Dict[str, Any]], clause: str) -> List[Dict[str, Any]]:
    items = [item.strip() for item in _split_by_comma(clause) if item.strip()]
    if not items:
        raise ValueError("ORDER BY clause is empty")

    orders: List[Tuple[str, bool]] = []
    for item in items:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_\.]*)(?:\s+(ASC|DESC))?$", item, re.IGNORECASE)
        if not m:
            raise ValueError(f"Invalid ORDER BY item: {item}")
        col = m.group(1)
        dir_token = (m.group(2) or "ASC").upper()
        asc = dir_token == "ASC"
        orders.append((col, asc))

    # Stable sort from last key to first
    sorted_rows = list(rows)
    for col, asc in reversed(orders):
        sorted_rows.sort(key=partial(_order_key_for_col, col=col, asc=asc), reverse=not asc)
    return sorted_rows


def _order_key_for_col(rec: Dict[str, Any], col: str, asc: bool) -> Tuple:
    val = _get_field(rec, col)
    return _safe_sort_key(val)


def _safe_sort_key(val: Any) -> Tuple[int, Any]:
    """
    Produce a comparable key for possibly heterogeneous values.
    Order within groups:
      numbers (0), strings (1), booleans (treated as numbers), None (2)
    """
    if val is None:
        return (3, 0)
    if isinstance(val, bool):
        return (0, int(val))
    if isinstance(val, (int, float)):
        return (0, val)
    return (1, str(val).lower())
