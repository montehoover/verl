import re
import operator
from functools import partial


def run_custom_query(dataset, query):
    """
    Execute a basic SQL-like statement on a dataset (list of dicts).
    Supports:
      - SELECT col1, col2 or SELECT *
      - WHERE with simple conditions combined by AND/OR
          Supported operators: =, ==, !=, <>, <, <=, >, >=, IN (...)
          Literals: numbers, booleans (true/false), null/none, and strings (quoted or unquoted words)
      - ORDER BY col [ASC|DESC] (supports multiple columns via comma separation)

    Args:
        dataset: list[dict]
        query: str

    Returns:
        list[dict]: query result rows

    Raises:
        ValueError: for parsing/processing errors
    """
    try:
        _validate_dataset(dataset)
        select_part, where_part, order_part = _parse_query_top(query)

        # SELECT
        select_fields = _parse_select(select_part)

        # WHERE
        predicate = (lambda row: True)
        if where_part is not None:
            predicate = _compile_where(where_part)

        # ORDER BY
        order_terms = []
        if order_part is not None:
            order_terms = _parse_order_by(order_part)

        # Filter
        filtered = [row for row in dataset if predicate(row)]

        # Order
        if order_terms:
            # To support different directions per column, perform stable sorts
            # from the last key to the first.
            for field, descending in reversed(order_terms):
                key_fn = _make_sort_key(field)
                # If any error due to incomparable types arises, fallback to string-based tie-breakers
                try:
                    filtered.sort(key=key_fn, reverse=descending)
                except TypeError:
                    # Fallback: coerce values to strings in the key to ensure comparability
                    def fallback_key(rec, _field=field):
                        v = rec.get(_field, None)
                        return (v is None, str(v))
                    filtered.sort(key=fallback_key, reverse=descending)

        # Project
        if select_fields is None:
            # SELECT * -> return shallow copies to avoid accidental external mutation
            result = [dict(row) for row in filtered]
        else:
            result = [{field: row.get(field, None) for field in select_fields} for row in filtered]

        return result

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to process query: {exc}") from exc


# --------------------- Internal Helpers --------------------- #

_SELECT_RE = re.compile(
    r"""
    ^\s*SELECT\s+
    (?P<select>.*?)
    (?=\s+WHERE\s+|\s+ORDER\s+BY\s+|$)
    (?:\s+WHERE\s+(?P<where>.*?))?
    (?:\s+ORDER\s+BY\s+(?P<order>.*))?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")


def _validate_dataset(dataset):
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of dictionaries")
    for i, row in enumerate(dataset):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset item at index {i} is not a dictionary")


def _parse_query_top(query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")
    m = _SELECT_RE.match(query)
    if not m:
        raise ValueError("Invalid query format. Expected: SELECT <columns> [WHERE <conditions>] [ORDER BY <fields>]")
    select_part = m.group("select").strip()
    where_part = m.group("where").strip() if m.group("where") is not None else None
    order_part = m.group("order").strip() if m.group("order") is not None else None
    return select_part, where_part, order_part


def _parse_select(select_part):
    if select_part == "*":
        return None
    if not select_part:
        raise ValueError("SELECT list cannot be empty")
    cols = [c.strip() for c in select_part.split(",")]
    if not all(c for c in cols):
        raise ValueError("SELECT list contains empty column name")
    for c in cols:
        if not _IDENTIFIER_RE.match(c):
            raise ValueError(f"Invalid column name in SELECT: {c!r}")
    # Deduplicate while preserving order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


_OPS = {
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _compile_where(where_part):
    """
    Compile WHERE clause into a predicate(row)->bool.
    Supports simple conditions joined by AND/OR (no parentheses).
    """
    if not where_part.strip():
        raise ValueError("WHERE clause is empty")

    or_groups = _split_logical(where_part, "OR")
    and_groups = [ _split_logical(group, "AND") for group in or_groups ]

    compiled_groups = []
    for group in and_groups:
        compiled_conds = []
        for cond in group:
            cond = cond.strip()
            if not cond:
                raise ValueError("Empty condition in WHERE clause")
            compiled_conds.append(_compile_condition(cond))
        # group evaluates to True if all conditions true
        compiled_groups.append(lambda row, conds=compiled_conds: all(c(row) for c in conds))

    # Overall True if any OR-group is True
    return lambda row: any(g(row) for g in compiled_groups)


def _split_logical(expr, keyword):
    # Split by keyword as a standalone word, case-insensitive.
    # This does not handle parentheses—intentionally simple.
    parts = re.split(rf"\s+{keyword}\s+", expr, flags=re.IGNORECASE)
    return [p for p in parts if p is not None and p != ""]


def _compile_condition(cond_str):
    # IN operator: field IN (v1, v2, ...)
    m_in = re.match(r"^\s*([A-Za-z_]\w*)\s+IN\s*\((.*)\)\s*$", cond_str, flags=re.IGNORECASE | re.DOTALL)
    if m_in:
        field = m_in.group(1)
        values_str = m_in.group(2)
        values = [_parse_literal(tok) for tok in _split_args(values_str)]
        value_set = set(values)
        return lambda row, f=field, vs=value_set: row.get(f, None) in vs

    # Comparison operators
    m = re.match(r"^\s*([A-Za-z_]\w*)\s*(=|==|!=|<>|<=|>=|<|>)\s*(.+?)\s*$", cond_str, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError(f"Invalid condition: {cond_str!r}")
    field, op_sym, raw_value = m.group(1), m.group(2), m.group(3)
    op_sym = op_sym.upper()
    if op_sym not in _OPS:
        raise ValueError(f"Unsupported operator in WHERE: {op_sym}")
    op_fn = _OPS[op_sym]
    literal = _parse_literal(raw_value)

    # Return predicate
    def predicate(row, f=field, fn=op_fn, lit=literal):
        left = row.get(f, None)
        if left is None and fn not in (operator.eq, operator.ne):
            # Comparisons with None other than equality are False
            return False
        try:
            # Allow numerical comparison if both are numbers
            if isinstance(left, (int, float)) and isinstance(lit, (int, float)):
                return fn(float(left), float(lit))
            return fn(left, lit)
        except Exception:
            # As a fallback, compare stringified
            return fn(str(left), str(lit))

    return predicate


def _split_args(s):
    """
    Split a comma-separated list, respecting quoted strings.
    E.g., "'a,b', 2, \"x\"" -> ["'a,b'", "2", "\"x\""]
    """
    parts = []
    buf = []
    in_quote = None
    escape = False
    for ch in s:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            buf.append(ch)  # keep the backslash in the token; literal parser will unescape
            escape = True
            continue
        if in_quote:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            buf.append(ch)
            continue
        if ch == "," and not in_quote:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    # Remove any empty entries caused by stray commas/spaces
    return [p for p in parts if p != ""]


def _parse_literal(token):
    """
    Parse a token into a Python literal:
      - quoted strings with ' or "
      - numbers (int/float)
      - booleans: true/false
      - null/none -> None
      - otherwise: bareword string
    """
    token = token.strip()
    if not token:
        return ""

    # Quoted string
    if (token[0] == token[-1]) and token[0] in ("'", '"'):
        inner = token[1:-1]
        # Unescape common sequences: \', \", \\
        inner = inner.replace(r"\\", "\\").replace(r"\'", "'").replace(r"\"", '"')
        return inner

    low = token.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None

    # Numeric
    try:
        if re.fullmatch(r"[-+]?\d+", token):
            return int(token)
        # float (supports exponent)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", token):
            return float(token)
        # Also allow pure exponent like 1e5 without decimal point
        if re.fullmatch(r"[-+]?\d+(?:[eE][-+]?\d+)", token):
            return float(token)
    except Exception:
        pass

    # Bareword -> string
    return token


def _make_sort_key(field):
    def key(row, f=field):
        v = row.get(f, None)
        return _sortable_value(v)
    return key


def _sortable_value(v):
    # Tuple to ensure different types are comparable
    # Order: None < bool < number < str < other
    if v is None:
        return (0, 0)
    if isinstance(v, bool):
        return (1, v)
    if isinstance(v, (int, float)):
        return (2, float(v))
    if isinstance(v, str):
        return (3, v)
    return (4, str(v))


def _parse_order_by(order_part):
    """
    Parse ORDER BY clause into a list of (field, descending) tuples.
    Accepts: "col", "col ASC", "col DESC", multiple terms separated by commas.
    """
    if not order_part.strip():
        raise ValueError("ORDER BY clause is empty")
    terms = []
    for token in _split_args(order_part):
        m = re.match(r"^\s*([A-Za-z_]\w*)(?:\s+(ASC|DESC))?\s*$", token, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Invalid ORDER BY term: {token!r}")
        field = m.group(1)
        direction = m.group(2)
        descending = bool(direction and direction.upper() == "DESC")
        terms.append((field, descending))
    if not terms:
        raise ValueError("ORDER BY must specify at least one column")
    return terms
