import re
import operator
from functools import partial


def run_sql_query(dataset, sql_query):
    """
    Execute a simple SQL-like query over a list of dictionaries.

    Supported clauses:
    - SELECT <columns>            (e.g., SELECT *, SELECT name, age)
    - WHERE <expression>          (comparisons with =, !=, <, <=, >, >=; logical AND, OR, NOT; parentheses)
    - ORDER BY <col> [ASC|DESC], ...

    Notes:
    - FROM is optional and ignored if present (since dataset is provided as an argument).
    - Column identifiers are simple alphanumeric/underscore names.
    - String literals can be single or double quoted and support simple backslash escapes.
    - Numbers: integers and floats.
    - Booleans: TRUE/FALSE (case-insensitive). NULL is supported.
    - Missing fields in SELECT projection produce None values.
    - Sorting across different types is supported with a consistent fallback.

    Args:
        dataset: List[Dict[str, Any]]
        sql_query: str

    Returns:
        List[Dict[str, Any]]

    Raises:
        ValueError: If the query is malformed or execution fails.
    """
    try:
        if not isinstance(dataset, list) or not all(isinstance(r, dict) for r in dataset):
            raise ValueError("dataset must be a list of dictionaries")
        if not isinstance(sql_query, str) or not sql_query.strip():
            raise ValueError("sql_query must be a non-empty string")

        parts = _parse_query(sql_query)
        select_cols = _parse_select(parts["select"])
        where_rpn = _parse_where(parts.get("where"))
        order_by = _parse_order_by(parts.get("order_by"))

        # Filter (WHERE)
        if where_rpn is not None:
            filtered = [row for row in dataset if _eval_where_rpn(where_rpn, row)]
        else:
            filtered = list(dataset)

        # Order (ORDER BY)
        if order_by:
            # Stable multi-key sort: apply from last to first
            for col, asc in reversed(order_by):
                filtered.sort(key=lambda r, c=col: _sort_key(r.get(c)), reverse=not asc)

        # Project (SELECT)
        result = []
        if select_cols == ["*"]:
            for row in filtered:
                # Return a shallow copy
                result.append(dict(row))
        else:
            for row in filtered:
                projected = {col: row.get(col, None) for col in select_cols}
                result.append(projected)

        return result

    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as exc:
        raise ValueError(f"Failed to execute query: {exc}") from None


# ------------------------ Parsing helpers ------------------------ #

_SELECT_PATTERN = re.compile(
    r"""
    ^\s*SELECT\s+(?P<select>.+?)                                   # SELECT list
    (?:\s+FROM\s+\S+)?                                             # optional FROM <ignored>
    (?:\s+WHERE\s+(?P<where>.+?))?                                 # optional WHERE
    (?:\s+ORDER\s+BY\s+(?P<order_by>.+))?                          # optional ORDER BY
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def _parse_query(query):
    m = _SELECT_PATTERN.match(query)
    if not m:
        raise ValueError("Malformed query. Expected: SELECT <cols> [WHERE <expr>] [ORDER BY <cols>]")
    select = m.group("select")
    where = m.group("where")
    order_by = m.group("order_by")
    return {"select": select, "where": where, "order_by": order_by}


def _parse_select(select_text):
    text = select_text.strip()
    if text == "*":
        return ["*"]
    cols = [c.strip() for c in text.split(",")]
    if not cols or any(not c for c in cols):
        raise ValueError("SELECT list is empty or malformed")
    ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for c in cols:
        if not ident_re.match(c):
            raise ValueError(f"Invalid column in SELECT list: {c}")
    return cols


def _parse_order_by(order_text):
    if not order_text:
        return []
    items = [p.strip() for p in order_text.split(",")]
    if not items:
        raise ValueError("ORDER BY clause is malformed")
    order = []
    item_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s+(ASC|DESC))?$", re.IGNORECASE)
    for item in items:
        m = item_re.match(item)
        if not m:
            raise ValueError(f"Invalid ORDER BY item: {item}")
        col = m.group(1)
        dir_token = (m.group(2) or "ASC").upper()
        asc = dir_token == "ASC"
        order.append((col, asc))
    return order


# ------------------------ WHERE parsing (tokenize -> RPN) ------------------------ #

_TOKEN_RE = re.compile(
    r"""
    \s*
    (
        (?P<STRING>'(?:\\.|[^\\'])*'|"(?:\\.|[^\\"])*") |
        (?P<NUMBER>\d+(?:\.\d+)?) |
        (?P<LPAREN>\() |
        (?P<RPAREN>\)) |
        (?P<OP><=|>=|!=|=|<|>) |
        (?P<WORD>[A-Za-z_][A-Za-z0-9_]*)
    )
    """,
    re.VERBOSE | re.DOTALL,
)

_LOGICAL = {"AND", "OR", "NOT"}
_LITERALS = {"TRUE": True, "FALSE": False, "NULL": None}


def _parse_where(where_text):
    if not where_text:
        return None
    tokens = _tokenize(where_text)
    if not tokens:
        raise ValueError("Empty WHERE expression")

    # Shunting-yard to produce RPN
    # Precedence: comparisons (3) > NOT (2) > AND (1) > OR (0)
    # NOT is unary and right-associative
    precedence = {
        "CMP": 3,  # any comparison operator
        "NOT": 2,
        "AND": 1,
        "OR": 0,
    }

    output = []
    ops = []

    def is_op(tok):
        return tok["type"] in ("CMP", "AND", "OR", "NOT")

    def prec(tok):
        if tok["type"] == "CMP":
            return precedence["CMP"]
        return precedence[tok["type"]]

    def is_right_assoc(tok):
        return tok["type"] == "NOT"

    prev_was_op_or_lparen = True  # For validating unary/binary placement

    for tok in tokens:
        ttype = tok["type"]

        if ttype in ("NUMBER", "STRING", "IDENT", "BOOLEAN", "NULL"):
            output.append(tok)
            prev_was_op_or_lparen = False

        elif ttype == "LPAREN":
            ops.append(tok)
            prev_was_op_or_lparen = True

        elif ttype == "RPAREN":
            # Pop until LPAREN
            while ops and ops[-1]["type"] != "LPAREN":
                output.append(ops.pop())
            if not ops or ops[-1]["type"] != "LPAREN":
                raise ValueError("Mismatched parentheses in WHERE clause")
            ops.pop()  # discard LPAREN
            prev_was_op_or_lparen = False

        elif is_op(tok):
            # Validate placement
            if tok["type"] == "CMP":
                if prev_was_op_or_lparen:
                    raise ValueError("Comparison operator in invalid position")
            elif tok["type"] in ("AND", "OR"):
                if prev_was_op_or_lparen:
                    raise ValueError(f"Logical operator {tok['type']} in invalid position")
            elif tok["type"] == "NOT":
                # NOT can be at start or after another op or '('
                if not prev_was_op_or_lparen:
                    # Allow "a = NOT b"? Not supporting; require NOT before an operand or '('
                    # If used after operand, it's invalid in this simplified grammar
                    raise ValueError("NOT must precede an expression or '('")

            # Pop stack based on precedence and associativity
            while ops and is_op(ops[-1]):
                top = ops[-1]
                if is_right_assoc(tok):
                    cond = prec(top) > prec(tok)
                else:
                    cond = prec(top) >= prec(tok)
                if cond:
                    output.append(ops.pop())
                else:
                    break
            ops.append(tok)
            prev_was_op_or_lparen = True if tok["type"] != "NOT" else True

        else:
            raise ValueError(f"Unexpected token in WHERE: {tok}")

    # Drain operator stack
    while ops:
        top = ops.pop()
        if top["type"] in ("LPAREN", "RPAREN"):
            raise ValueError("Mismatched parentheses in WHERE clause")
        output.append(top)

    return output


def _tokenize(text):
    tokens = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if not m:
            # Skip pure whitespace; otherwise error
            if text[pos].isspace():
                pos += 1
                continue
            snippet = text[pos : min(n, pos + 20)]
            raise ValueError(f"Invalid token in WHERE near: {snippet!r}")
        pos = m.end()
        # Determine which named group matched
        if m.lastgroup == "STRING":
            raw = m.group(m.lastgroup)
            tokens.append({"type": "STRING", "value": _unquote_string(raw)})
        elif m.lastgroup == "NUMBER":
            num_str = m.group(m.lastgroup)
            value = int(num_str) if "." not in num_str else float(num_str)
            tokens.append({"type": "NUMBER", "value": value})
        elif m.lastgroup == "LPAREN":
            tokens.append({"type": "LPAREN"})
        elif m.lastgroup == "RPAREN":
            tokens.append({"type": "RPAREN"})
        elif m.lastgroup == "OP":
            tokens.append({"type": "CMP", "op": m.group(m.lastgroup)})
        elif m.lastgroup == "WORD":
            word = m.group(m.lastgroup)
            upper = word.upper()
            if upper in _LOGICAL:
                tokens.append({"type": upper})
            elif upper in _LITERALS:
                val = _LITERALS[upper]
                if val is True or val is False:
                    tokens.append({"type": "BOOLEAN", "value": val})
                else:
                    tokens.append({"type": "NULL", "value": None})
            else:
                tokens.append({"type": "IDENT", "value": word})
        else:
            raise ValueError("Unrecognized token")
    return tokens


def _unquote_string(s):
    # s starts and ends with the same quote type
    if len(s) < 2 or s[0] != s[-1] or s[0] not in ("'", '"'):
        raise ValueError(f"Invalid string literal: {s}")
    body = s[1:-1]

    def repl(m):
        ch = m.group(1)
        return {
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "\\": "\\",
            "'": "'",
            '"': '"',
        }.get(ch, ch)

    # Replace backslash escapes
    body = re.sub(r"\\(.)", repl, body)
    return body


# ------------------------ WHERE evaluation ------------------------ #

def _eval_where_rpn(rpn, row):
    stack = []
    for tok in rpn:
        ttype = tok["type"]
        if ttype in ("NUMBER", "STRING", "BOOLEAN", "NULL"):
            stack.append(tok.get("value"))
        elif ttype == "IDENT":
            stack.append(row.get(tok["value"]))
        elif ttype == "NOT":
            if not stack:
                raise ValueError("NOT missing operand")
            a = stack.pop()
            stack.append(not _truthy(a))
        elif ttype == "AND":
            if len(stack) < 2:
                raise ValueError("AND missing operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(_truthy(a) and _truthy(b))
        elif ttype == "OR":
            if len(stack) < 2:
                raise ValueError("OR missing operands")
            b = stack.pop()
            a = stack.pop()
            stack.append(_truthy(a) or _truthy(b))
        elif ttype == "CMP":
            if len(stack) < 2:
                raise ValueError("Comparison missing operands")
            b = stack.pop()
            a = stack.pop()
            comp_fn = _comp_func(tok["op"])
            stack.append(comp_fn(a, b))
        else:
            raise ValueError(f"Unsupported token during evaluation: {tok}")
    if len(stack) != 1:
        raise ValueError("Invalid WHERE expression")
    return _truthy(stack[0])


def _truthy(v):
    return bool(v)


def _comp_func(op_symbol):
    # Use functools.partial with operator functions and our safe wrapper
    mapping = {
        "=": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }
    if op_symbol not in mapping:
        raise ValueError(f"Unsupported comparison operator: {op_symbol}")
    return partial(_safe_compare, mapping[op_symbol])


def _coerce_numbers(a, b):
    # Try to coerce comparable values
    def to_num(x):
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            try:
                return float(x)
            except ValueError:
                return None
        return None

    na = to_num(a)
    nb = to_num(b)
    if na is not None and nb is not None:
        return na, nb
    return None, None


def _safe_compare(op_func, a, b):
    # Handle None semantics and type mismatches gracefully
    if a is None or b is None:
        if op_func is operator.eq:
            return a is None and b is None
        if op_func is operator.ne:
            return (a is None) != (b is None) or (a is not None and b is None) or (a is None and b is not None)
        # Ordering comparisons with NULL -> False
        return False

    # Try numeric comparison if possible
    na, nb = _coerce_numbers(a, b)
    if na is not None and nb is not None:
        try:
            return op_func(na, nb)
        except Exception:
            pass

    # Fall back to direct comparison
    try:
        return op_func(a, b)
    except Exception:
        # Cross-type fallback: compare by string representation
        return op_func(str(a), str(b))


# ------------------------ Sorting helpers ------------------------ #

def _sort_key(v):
    # Normalize values to a comparable key across types
    # Order within type is preserved; types are grouped deterministically.
    # None is treated separately to allow consistent placement.
    if v is None:
        return (1, "NONE", 0)
    # Normalize booleans before ints (since bool is subclass of int)
    if isinstance(v, bool):
        return (0, "BOOL", 1 if v else 0)
    if isinstance(v, (int, float)):
        return (0, "NUM", float(v))
    if isinstance(v, str):
        return (0, "STR", v)
    # Fallback for other types
    return (0, "OTHER", str(v))
