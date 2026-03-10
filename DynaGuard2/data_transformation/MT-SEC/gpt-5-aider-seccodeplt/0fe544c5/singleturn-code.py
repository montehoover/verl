import re
import operator
from functools import partial


class _ParseError(ValueError):
    pass


def handle_sql_query(records, sql_command):
    """
    Execute a simplified SQL-like query on a list of dictionary records.

    Supported syntax (case-insensitive keywords):
      SELECT <columns> FROM <identifier> [WHERE <condition>] [ORDER BY <field> [ASC|DESC]]

    - <columns>: '*' or comma-separated identifiers
    - <identifier>: dataset/table name (ignored; must be present)
    - <condition>: comparisons combined with AND/OR/NOT and parentheses
        comparisons supported: =, ==, !=, <>, >, >=, <, <=
        literals supported: numbers (int/float) and quoted strings ('...' or "...")
        field references are bare identifiers
    - ORDER BY: one field, optional ASC/DESC (default ASC)

    Args:
      records (list[dict]): input data
      sql_command (str): query string

    Returns:
      list[dict]: query results

    Raises:
      ValueError: if the query is malformed or evaluation fails
    """
    try:
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise _ParseError("records must be a list of dictionaries")

        if not isinstance(sql_command, str) or not sql_command.strip():
            raise _ParseError("sql_command must be a non-empty string")

        parsed = _parse_query(sql_command)

        # Build WHERE predicate
        where_clause = parsed.get("where")
        if where_clause:
            predicate = _compile_where(where_clause)
        else:
            predicate = lambda _rec: True  # accept all

        # Filter records
        try:
            filtered = [rec for rec in records if predicate(rec)]
        except Exception as e:
            raise _ParseError(f"Failed to evaluate WHERE clause: {e}") from e

        # ORDER BY
        order_by = parsed.get("order_by")
        if order_by:
            field, ascending = order_by
            try:
                filtered = sorted(filtered, key=_make_sort_key(field), reverse=not ascending)
            except TypeError as e:
                raise _ParseError(f"Failed to sort by '{field}': {e}") from e

        # SELECT projection
        select_cols = parsed["select"]
        if select_cols == "*":
            # Return shallow copies to prevent accidental mutation of original input
            result = [dict(rec) for rec in filtered]
        else:
            result = [{col: rec.get(col, None) for col in select_cols} for rec in filtered]

        return result
    except _ParseError as e:
        raise ValueError(str(e)) from None
    except ValueError:
        # Re-raise explicit ValueErrors (e.g., from numeric conversions) as-is to conform to spec
        raise
    except Exception as e:
        # Any other unexpected error becomes a ValueError per spec
        raise ValueError(f"Query execution failed: {e}") from None


# ---------------------------
# Parsing and evaluation utils
# ---------------------------

_SELECT_RE = re.compile(
    r"""
    ^\s*SELECT\s+(?P<select>.+?)\s+
    FROM\s+(?P<from>\w+)
    (?:\s+WHERE\s+(?P<where>.+?))?
    (?:\s+ORDER\s+BY\s+(?P<order_by>.+))?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_query(sql):
    m = _SELECT_RE.match(sql)
    if not m:
        raise _ParseError("Query must be of the form: SELECT <columns> FROM <name> [WHERE ...] [ORDER BY ...]")

    select_raw = m.group("select")
    from_name = m.group("from")  # present but currently unused
    where_raw = m.group("where")
    order_raw = m.group("order_by")

    # Parse SELECT list
    select = _parse_select_list(select_raw)

    # Parse ORDER BY
    order_by = _parse_order_by(order_raw) if order_raw else None

    return {"select": select, "from": from_name, "where": where_raw, "order_by": order_by}


def _parse_select_list(text):
    text = text.strip()
    if text == "*":
        return "*"
    cols = [c.strip() for c in text.split(",")]
    if not cols or any(not c or not _IDENT_RE.match(c) for c in cols):
        raise _ParseError("SELECT list must be '*' or a comma-separated list of identifiers")
    # normalize column names as provided (case-sensitive)
    return cols


def _parse_order_by(text):
    # Support single column, optional ASC/DESC
    if text is None:
        return None
    s = text.strip()
    # If multiple columns are provided, we consider this malformed for this simplified engine
    # but we try to parse the first valid segment like: "col", "col ASC", "col DESC"
    parts = s.split(",")
    if len(parts) > 1:
        raise _ParseError("Only a single ORDER BY field is supported")
    token_str = parts[0].strip()
    # Split by whitespace to capture optional direction
    tokens = token_str.split()
    if not tokens:
        raise _ParseError("ORDER BY clause is empty")
    field = tokens[0]
    if not _IDENT_RE.match(field):
        raise _ParseError("ORDER BY field must be a valid identifier")
    direction = "ASC"
    if len(tokens) > 1:
        dir_token = tokens[1].upper()
        if dir_token not in ("ASC", "DESC"):
            raise _ParseError("ORDER BY direction must be ASC or DESC")
        direction = dir_token
    if len(tokens) > 2:
        raise _ParseError("ORDER BY supports only one field and an optional direction")
    return (field, direction == "ASC")


# WHERE clause parsing

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<LPAREN>\() |
        (?P<RPAREN>\)) |
        (?P<OP><=|>=|<>|!=|==|=|<|>) |
        (?P<STRING>'([^'\\]|\\.)*'|"([^"\\]|\\.)*") |
        (?P<NUMBER>\d+\.\d+|\d+) |
        (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*) |
        (?P<UNKNOWN>.)
    )
    """,
    re.VERBOSE | re.DOTALL,
)

# Operator metadata
_COMPARATORS = {
    "=": operator.eq,
    "==": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}
_BOOL_PRECEDENCE = {
    # higher value = higher precedence
    # Comparators produce predicates; they bind tighter than NOT/AND/OR
    "CMP": 4,       # any comparator
    "NOT": 3,
    "AND": 2,
    "OR": 1,
}
_LEFT_ASSOC = {"CMP", "AND", "OR"}  # NOT is right-associative


def _unescape_quoted(s):
    # remove surrounding quotes and unescape \" or \'
    quote = s[0]
    inner = s[1:-1]
    # replace backslash-escaped characters with the char itself
    return re.sub(r"\\(.)", r"\1", inner)


def _tokenize_where(s):
    tokens = []
    pos = 0
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise _ParseError(f"Invalid token near: {s[pos:pos+20]!r}")
        pos = m.end()
        kind = m.lastgroup
        lexeme = m.group(kind)
        if kind in ("LPAREN", "RPAREN"):
            tokens.append((kind, lexeme))
        elif kind == "OP":
            tokens.append(("OP", lexeme))
        elif kind == "STRING":
            val = _unescape_quoted(lexeme)
            tokens.append(("LIT", val))
        elif kind == "NUMBER":
            if "." in lexeme:
                try:
                    num = float(lexeme)
                except ValueError:
                    raise _ParseError(f"Invalid number: {lexeme}")
            else:
                try:
                    num = int(lexeme)
                except ValueError:
                    raise _ParseError(f"Invalid number: {lexeme}")
            tokens.append(("LIT", num))
        elif kind == "IDENT":
            upper = lexeme.upper()
            if upper in ("AND", "OR", "NOT"):
                tokens.append(("BOOL", upper))
            else:
                tokens.append(("FIELD", lexeme))
        elif kind == "UNKNOWN":
            raise _ParseError(f"Unexpected character in WHERE: {lexeme!r}")
        else:
            raise _ParseError("Unknown token type")
    return tokens


def _to_rpn(tokens):
    """
    Shunting-yard algorithm to produce Reverse Polish Notation for:
      - operands: FIELD/LIT
      - comparison ops: OP (-> 'CMP' precedence)
      - boolean ops: BOOL (AND/OR/NOT)
    """
    output = []
    ops = []

    def _prec(tok):
        if tok[0] == "OP":
            return _BOOL_PRECEDENCE["CMP"]
        if tok[0] == "BOOL":
            return _BOOL_PRECEDENCE[tok[1]]
        return -1

    def _is_left_assoc(tok):
        if tok[0] == "OP":
            return True
        if tok[0] == "BOOL":
            return tok[1] in _LEFT_ASSOC
        return True

    for tok in tokens:
        ttype, tval = tok
        if ttype in ("FIELD", "LIT"):
            output.append(tok)
        elif ttype == "OP" or ttype == "BOOL":
            while ops:
                top = ops[-1]
                if top[0] == "LPAREN":
                    break
                if (_is_left_assoc(tok) and _prec(tok) <= _prec(top)) or (not _is_left_assoc(tok) and _prec(tok) < _prec(top)):
                    output.append(ops.pop())
                else:
                    break
            ops.append(tok)
        elif ttype == "LPAREN":
            ops.append(tok)
        elif ttype == "RPAREN":
            # Pop until LPAREN
            while ops and ops[-1][0] != "LPAREN":
                output.append(ops.pop())
            if not ops or ops[-1][0] != "LPAREN":
                raise _ParseError("Mismatched parentheses in WHERE clause")
            ops.pop()  # discard LPAREN
        else:
            raise _ParseError(f"Unexpected token: {tok}")

    while ops:
        if ops[-1][0] in ("LPAREN", "RPAREN"):
            raise _ParseError("Mismatched parentheses in WHERE clause")
        output.append(ops.pop())

    return output


def _resolve_operand(operand, record):
    ttype, tval = operand
    if ttype == "FIELD":
        return record.get(tval, None)
    elif ttype == "LIT":
        return tval
    else:
        raise _ParseError(f"Invalid operand in expression: {operand}")


def _compile_where(where_str):
    tokens = _tokenize_where(where_str)
    if not tokens:
        raise _ParseError("WHERE clause is empty")

    rpn = _to_rpn(tokens)

    stack = []

    for tok in rpn:
        ttype, tval = tok
        if ttype in ("FIELD", "LIT"):
            stack.append(tok)
        elif ttype == "OP":
            # comparator requires two operands -> becomes a predicate
            if tval not in _COMPARATORS:
                raise _ParseError(f"Unsupported operator: {tval}")
            if len(stack) < 2:
                raise _ParseError("Invalid comparison expression")
            right = stack.pop()
            left = stack.pop()
            comp = _COMPARATORS[tval]

            def make_pred(lop, rop, cmpop):
                return lambda rec: _safe_compare(_resolve_operand(lop, rec), _resolve_operand(rop, rec), cmpop)

            pred = make_pred(left, right, comp)
            stack.append(("PRED", pred))
        elif ttype == "BOOL":
            if tval == "NOT":
                if not stack:
                    raise _ParseError("NOT missing operand")
                node = stack.pop()
                # node can be PRED or raw operand -> normalize to predicate
                pred = _ensure_predicate(node)
                stack.append(("PRED", lambda rec, p=pred: not p(rec)))
            elif tval in ("AND", "OR"):
                if len(stack) < 2:
                    raise _ParseError(f"{tval} requires two operands")
                right = stack.pop()
                left = stack.pop()
                p_right = _ensure_predicate(right)
                p_left = _ensure_predicate(left)
                if tval == "AND":
                    stack.append(("PRED", lambda rec, a=p_left, b=p_right: a(rec) and b(rec)))
                else:
                    stack.append(("PRED", lambda rec, a=p_left, b=p_right: a(rec) or b(rec)))
            else:
                raise _ParseError(f"Unknown boolean operator: {tval}")
        else:
            raise _ParseError(f"Unexpected token in RPN: {tok}")

    if len(stack) != 1 or stack[0][0] != "PRED":
        raise _ParseError("WHERE clause did not resolve to a boolean expression")

    return stack[0][1]


def _ensure_predicate(node):
    ttype, tval = node
    if ttype == "PRED":
        return tval
    elif ttype in ("FIELD", "LIT"):
        # Interpret a bare operand as truthiness check (non-zero/ non-empty)
        # This is not standard SQL, but avoids cryptic errors if user writes: WHERE someField
        return lambda rec, n=node: bool(_resolve_operand(n, rec))
    else:
        raise _ParseError("Invalid expression element in boolean context")


def _safe_compare(a, b, cmpop):
    # SQL-like behavior: comparisons with None return False (except equality with None, which is False too in SQL unless IS NULL)
    if a is None or b is None:
        # We treat all direct comparisons with None as False for simplicity
        return False
    try:
        return bool(cmpop(a, b))
    except Exception:
        # Attempt numeric coercion if possible
        try:
            a_num = float(a) if not isinstance(a, bool) else int(a)
            b_num = float(b) if not isinstance(b, bool) else int(b)
            return bool(cmpop(a_num, b_num))
        except Exception:
            # Fallback to string comparison
            try:
                return bool(cmpop(str(a), str(b)))
            except Exception:
                return False


def _make_sort_key(field):
    # Return a key function that sorts None last and avoids TypeError for mixed types.
    def key(rec):
        v = rec.get(field, None)
        if v is None:
            return (1, 0, "")  # Nones last
        # type rank: numbers < strings < bools < others
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return (0, 0, v)
        if isinstance(v, str):
            return (0, 1, v)
        if isinstance(v, bool):
            return (0, 2, int(v))
        # Fallback to string representation
        return (0, 3, str(v))
    return key
