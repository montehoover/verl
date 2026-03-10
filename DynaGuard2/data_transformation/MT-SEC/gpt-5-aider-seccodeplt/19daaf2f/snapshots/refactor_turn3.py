import re
import operator
import logging
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
    logger = _get_logger()
    try:
        if not isinstance(dataset, list) or not all(isinstance(r, dict) for r in dataset):
            raise ValueError("dataset must be a list of dictionaries")
        if not isinstance(sql_query, str) or not sql_query.strip():
            raise ValueError("sql_query must be a non-empty string")

        logger.info("Received SQL-like query (records=%d): %s", len(dataset), sql_query)

        parsed = parse_sql_query(sql_query)

        # Log operations to be performed (human-readable)
        select_cols = parsed["select_cols"]
        where_text = (parsed.get("raw_parts") or {}).get("where")
        order_by = parsed["order_by"]
        order_text = ", ".join(f"{col} {'ASC' if asc else 'DESC'}" for col, asc in order_by) if order_by else None

        logger.info(
            "Operations:"
            "\n - SELECT: %s"
            "\n - WHERE: %s"
            "\n - ORDER BY: %s",
            "*" if select_cols == ["*"] else ", ".join(select_cols),
            where_text if where_text else "None",
            order_text if order_text else "None",
        )

        result = execute_parsed_query(dataset, parsed)

        # Log final result summary with a preview
        preview = _format_rows_preview(result, max_rows=10, max_chars=2000)
        logger.info("Query result: %d row(s)\n%s", len(result), preview)

        return result

    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as exc:
        try:
            logger.exception("Failed to execute query: %s", sql_query)
        except Exception:
            # If logger setup failed for some reason, fall back to raising only
            pass
        raise ValueError(f"Failed to execute query: {exc}") from None


# ------------------------ Logging helpers ------------------------ #

def _get_logger():
    """
    Initialize and return a logger for SQL-like query execution.
    Ensures a human-readable console handler is attached exactly once.
    """
    logger = logging.getLogger("sql_query")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def _format_rows_preview(rows, max_rows=10, max_chars=2000):
    """
    Return a human-readable preview string of rows (list of dicts),
    limited by max_rows and max_chars.
    """
    try:
        subset = rows[:max_rows]
    except Exception:
        return "<unprintable result>"

    def safe_repr(obj):
        try:
            return repr(obj)
        except Exception:
            try:
                return str(obj)
            except Exception:
                return "<unprintable>"

    content = "[\n  " + ",\n  ".join(safe_repr(r) for r in subset) + ("\n]" if subset else "]")
    if len(rows) > max_rows:
        content += f"\n... and {len(rows) - max_rows} more row(s)"
    if len(content) > max_chars:
        content = content[:max_chars] + " ... [truncated]"
    return content


# ------------------------ Pipeline: parse -> execute ------------------------ #

def parse_sql_query(sql_query):
    """
    Pure function: Parse SQL-like query string into an execution plan.
    Returns a dict: {
        select_cols: List[str],
        where_rpn: Optional[List],
        order_by: List[Tuple[str,bool]],
        raw_parts: Dict[str, Optional[str]]  # echoes original text parts for logging
    }
    """
    parts = _parse_query(sql_query)
    select_cols = _parse_select(parts["select"])
    where_rpn = _parse_where(parts.get("where"))
    order_by = _parse_order_by(parts.get("order_by"))
    return {
        "select_cols": select_cols,
        "where_rpn": where_rpn,
        "order_by": order_by,
        "raw_parts": {
            "select": parts["select"],
            "where": parts.get("where"),
            "order_by": parts.get("order_by"),
        },
    }


def execute_parsed_query(dataset, parsed):
    """
    Pure function: Execute a parsed query plan against dataset and return results.
    Does not mutate the input dataset or its records.
    """
    if not isinstance(dataset, list) or not all(isinstance(r, dict) for r in dataset):
        raise ValueError("dataset must be a list of dictionaries")

    select_cols = parsed["select_cols"]
    where_rpn = parsed["where_rpn"]
    order_by = parsed["order_by"]

    # Build pipeline stages
    stages = []
    if where_rpn is not None:
        stages.append(_stage_filter(where_rpn))
    if order_by:
        stages.append(_stage_order(order_by))
    stages.append(_stage_project(select_cols))

    # Execute pipeline
    data = list(dataset)  # work on a shallow copy of the list
    for stage in stages:
        data = stage(data)
    return data


def _stage_filter(where_rpn):
    def stage(data):
        return [row for row in data if _eval_where_rpn(where_rpn, row)]
    return stage


def _stage_order(order_by):
    def stage(data):
        out = list(data)
        # Stable multi-key sort: apply from last to first
        for col, asc in reversed(order_by):
            out = sorted(out, key=lambda r, c=col: _sort_key(r.get(c)), reverse=not asc)
        return out
    return stage


def _stage_project(select_cols):
    def stage(data):
        if select_cols == ["*"]:
            return [dict(row) for row in data]
        return [{col: row.get(col, None) for col in select_cols} for row in data]
    return stage


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
            prev_was_op_or_lparen = True

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
