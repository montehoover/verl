from collections import defaultdict
from typing import List, Dict, Any, Callable
import re
import operator
from functools import partial


def select_fields(records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """
    Extract specific fields from a list of record dictionaries.

    - records: List of dictionaries representing data records.
    - fields: List of field names to extract.

    Returns a new list of dictionaries that contain only the requested fields.
    Missing fields are included with a value of None.
    """
    result: List[Dict[str, Any]] = []

    if not records:
        return result

    for rec in records:
        # Ensure we have a dictionary; if not, treat as empty
        base = rec if isinstance(rec, dict) else {}
        dd = defaultdict(lambda: None, base)
        extracted = {f: dd[f] for f in fields}
        result.append(extracted)

    return result


def filter_data(records: List[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Filter records by a provided condition function.

    - records: List of dictionaries representing data records.
    - condition: Callable that takes a single record (dict) and returns True if the record
      should be included, or False otherwise.

    Returns a new list containing only the records that satisfy the condition.
    """
    if not records:
        return []
    if not callable(condition):
        raise TypeError("condition must be a callable that accepts a record dictionary and returns a boolean")

    return [rec for rec in records if condition(rec)]


# ----------------------------
# SQL-like query support
# ----------------------------

_LOGICAL_TOKENS = {"AND", "OR", "NOT"}

_CMP_MAP = {
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _split_csv(expr: str) -> List[str]:
    """
    Split a comma-separated list while respecting quotes and parentheses.
    """
    parts: List[str] = []
    buf: List[str] = []
    in_single = False
    in_double = False
    depth = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
        elif ch == "\\" and (in_single or in_double) and i + 1 < len(expr):
            # keep escaped char inside strings
            buf.append(ch)
            i += 1
            buf.append(expr[i])
        elif ch == "(" and not in_single and not in_double:
            depth += 1
            buf.append(ch)
        elif ch == ")" and not in_single and not in_double:
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and not in_single and not in_double and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _unescape_string(s: str) -> str:
    s = s.replace("\\'", "'").replace('\\"', '"').replace("\\\\", "\\")
    s = s.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
    return s


def _parse_literal(token: str) -> Any:
    if token.lower() == "true":
        return True
    if token.lower() == "false":
        return False
    if token.lower() == "none" or token.lower() == "null":
        return None
    # number?
    try:
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token)
        if re.fullmatch(r"[+-]?\d*\.\d+(e[+-]?\d+)?", token, flags=re.IGNORECASE) or re.fullmatch(r"[+-]?\d+e[+-]?\d+", token, flags=re.IGNORECASE):
            return float(token)
    except Exception:
        pass
    # string literal with quotes
    if (len(token) >= 2) and ((token[0] == token[-1] == "'") or (token[0] == token[-1] == '"')):
        return _unescape_string(token[1:-1])
    # otherwise treat as identifier (field reference marker)
    return {"__field__": token}


def _tokenize_where(expr: str) -> List[Any]:
    tokens: List[Any] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue
        # operators two-char
        if i + 1 < n and expr[i:i+2] in ("<=", ">=", "!=", "==", "<>"):
            tokens.append(expr[i:i+2])
            i += 2
            continue
        if ch in "<>=":
            tokens.append(ch)
            i += 1
            continue
        # string literal
        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            buf: List[str] = [ch]
            while j < n:
                cj = expr[j]
                buf.append(cj)
                if cj == "\\" and j + 1 < n:
                    j += 2
                    continue
                if cj == quote:
                    j += 1
                    break
                j += 1
            if buf[-1] != quote:
                raise ValueError("Unterminated string literal in WHERE clause")
            token = "".join(buf)
            tokens.append(token)
            i = j
            continue
        # identifier or logical or number
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", expr[i:])
        if m:
            word = m.group(0)
            upper = word.upper()
            if upper in _LOGICAL_TOKENS:
                tokens.append(upper)
            else:
                tokens.append(word)
            i += len(word)
            continue
        # number literals starting with digit or sign
        m2 = re.match(r"[+-]?\d+(\.\d+)?(e[+-]?\d+)?", expr[i:], flags=re.IGNORECASE)
        if m2:
            tokens.append(m2.group(0))
            i += len(m2.group(0))
            continue
        raise ValueError(f"Unexpected token near: {expr[i:i+10]!r}")
    return tokens


def _to_rpn(tokens: List[Any]) -> List[Any]:
    """
    Convert tokens to Reverse Polish Notation (Shunting Yard).
    Comparison operators have highest precedence, then NOT, AND, OR.
    """
    out: List[Any] = []
    stack: List[Any] = []

    def prec(tok: Any) -> int:
        if tok in _CMP_MAP:
            return 4
        if tok == "NOT":
            return 3
        if tok == "AND":
            return 2
        if tok == "OR":
            return 1
        return 0

    def is_op(tok: Any) -> bool:
        return tok in _CMP_MAP or tok in _LOGICAL_TOKENS

    for tok in tokens:
        if tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                out.append(stack.pop())
            if not stack:
                raise ValueError("Mismatched parentheses in WHERE clause")
            stack.pop()
        elif is_op(tok):
            # For right-assoc NOT, treat as higher precedence but also pop strictly greater
            while stack and is_op(stack[-1]) and ((prec(stack[-1]) > prec(tok)) or (prec(stack[-1]) == prec(tok) and tok != "NOT")):
                out.append(stack.pop())
            stack.append(tok)
        else:
            out.append(tok)
    while stack:
        top = stack.pop()
        if top in ("(", ")"):
            raise ValueError("Mismatched parentheses in WHERE clause")
        out.append(top)
    return out


def _eval_rpn(rpn: List[Any], record: Dict[str, Any]) -> bool:
    def get_value(token: Any) -> Any:
        # token may be operator, identifier, number/string literal
        if isinstance(token, str) and token in _CMP_MAP:
            return token
        if isinstance(token, str) and token in _LOGICAL_TOKENS:
            return token
        # literals or identifiers
        lit = _parse_literal(token) if isinstance(token, str) else token
        if isinstance(lit, dict) and "__field__" in lit:
            name = lit["__field__"]
            return record.get(name, None)
        return lit

    stack: List[Any] = []
    for tok in rpn:
        if isinstance(tok, str) and tok in _LOGICAL_TOKENS:
            if tok == "NOT":
                if not stack:
                    raise ValueError("Malformed WHERE clause: NOT missing operand")
                a = stack.pop()
                stack.append(not bool(a))
            else:
                if len(stack) < 2:
                    raise ValueError("Malformed WHERE clause: logical operator missing operands")
                b = bool(stack.pop())
                a = bool(stack.pop())
                if tok == "AND":
                    stack.append(a and b)
                elif tok == "OR":
                    stack.append(a or b)
        elif isinstance(tok, str) and tok in _CMP_MAP:
            if len(stack) < 2:
                raise ValueError("Malformed WHERE clause: comparison missing operands")
            right_tok = stack.pop()
            left_tok = stack.pop()
            left = left_tok
            right = right_tok
            op = _CMP_MAP[tok]
            # Graceful handling for None
            if left is None or right is None:
                if op is operator.eq:
                    stack.append(left is right)
                elif op is operator.ne:
                    stack.append(left is not right)
                else:
                    stack.append(False)
            else:
                try:
                    stack.append(op(left, right))
                except Exception:
                    stack.append(False)
        else:
            stack.append(get_value(tok))
    if len(stack) != 1:
        raise ValueError("Malformed WHERE clause")
    return bool(stack[0])


def _normalize_for_sort(v: Any) -> Any:
    # Returns a tuple (rank, normalized_value) so cross-type comparisons won't crash.
    if v is None:
        return (3, 0)
    if isinstance(v, (int, float)):
        return (0, float(v))
    if isinstance(v, str):
        return (1, v)
    # booleans after strings
    if isinstance(v, bool):
        return (2, int(v))
    return (4, repr(v))


def run_custom_query(dataset: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """
    Execute a minimal SQL-like query supporting:
    - SELECT field1, field2 or SELECT *
    - optional WHERE with comparisons (=, ==, !=, <, <=, >, >=) combined using AND, OR, NOT
    - optional ORDER BY field [ASC|DESC][, field2 [ASC|DESC], ...]

    Raises ValueError if the query is malformed.
    """
    if not isinstance(dataset, list):
        raise ValueError("dataset must be a list of dictionaries")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    pattern = re.compile(
        r"^\s*SELECT\s+(?P<select>.+?)\s*(?:FROM\s+(?P<from>\S+)\s*)?(?:WHERE\s+(?P<where>.+?))?(?:\s+ORDER\s+BY\s+(?P<order>.+))?\s*$",
        flags=re.IGNORECASE | re.DOTALL,
    )
    m = pattern.match(query)
    if not m:
        raise ValueError("Malformed query: could not parse")
    select_part = m.group("select")
    where_part = m.group("where")
    order_part = m.group("order")

    # Parse SELECT
    select_fields_part: List[str] | None
    if select_part.strip() == "*":
        select_fields_part = None
    else:
        fields_list = _split_csv(select_part)
        if not fields_list:
            raise ValueError("Malformed SELECT clause")
        select_fields_part = [f.strip() for f in fields_list if f.strip()]
        if not select_fields_part:
            raise ValueError("Malformed SELECT clause")

    # Parse WHERE
    where_rpn = None
    if where_part:
        try:
            tokens = _tokenize_where(where_part)
            # Map lowercase logical tokens to uppercase for consistency
            tokens = [tok.upper() if isinstance(tok, str) and tok.upper() in _LOGICAL_TOKENS else tok for tok in tokens]
            where_rpn = _to_rpn(tokens)
        except ValueError as e:
            raise ValueError(f"Malformed WHERE clause: {e}") from e

    # Parse ORDER BY
    order_by: List[tuple[str, bool]] = []
    if order_part:
        parts = _split_csv(order_part)
        if not parts:
            raise ValueError("Malformed ORDER BY clause")
        for p in parts:
            seg = p.strip()
            if not seg:
                continue
            m_order = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s+(ASC|DESC))?$", seg, flags=re.IGNORECASE)
            if not m_order:
                raise ValueError(f"Malformed ORDER BY element: {seg!r}")
            field = m_order.group(1)
            direction = m_order.group(2)
            asc = True if not direction else direction.upper() == "ASC"
            order_by.append((field, asc))

    # Start processing
    # Shallow copy records (ensure dicts)
    records: List[Dict[str, Any]] = []
    for r in dataset:
        if isinstance(r, dict):
            records.append(r)
        else:
            # non-dict records are ignored gracefully
            continue

    # WHERE filter
    if where_rpn is not None:
        try:
            records = [r for r in records if _eval_rpn(where_rpn, r)]
        except ValueError as e:
            raise ValueError(f"Failed to evaluate WHERE: {e}") from e

    # ORDER BY (stable multi-key)
    if order_by:
        for field, asc in reversed(order_by):
            records.sort(key=lambda rec: _normalize_for_sort(rec.get(field, None)), reverse=not asc)

    # SELECT projection
    if select_fields_part is None:
        result = [dict(r) for r in records]
    else:
        # reuse select_fields to handle missing values as None
        result = select_fields(records, select_fields_part)

    return result
