from collections import defaultdict
from typing import Any, Dict, Iterable, List
from typing import Callable, List, Dict
import re
import operator
from functools import partial

def select_columns(records: List[Dict[str, Any]], columns: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Extract only the specified columns from each record.
    Missing columns are included with a value of None.
    """
    if not records:
        return []

    cols = list(columns)
    result: List[Dict[str, Any]] = []

    for record in records:
        safe_record = defaultdict(lambda: None, record or {})
        selected = {col: safe_record[col] for col in cols}
        result.append(selected)

    return result


def apply_filter(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Return only records that satisfy the given condition.
    The condition can be any callable that accepts a record and returns a bool.
    Logical operations can be composed in the callable, e.g.:
        lambda r: cond1(r) and (cond2(r) or not cond3(r))
    """
    if not records:
        return []

    return [record for record in records if condition(record)]


def handle_sql_query(records: List[Dict[str, Any]], sql_command: str) -> List[Dict[str, Any]]:
    """
    Execute a simple SQL-like query on in-memory records.

    Supported syntax (case-insensitive):
      SELECT <col1, col2, ... | *>
      [WHERE <expression>]
      [ORDER BY col1 [ASC|DESC], col2 [ASC|DESC], ...]

    WHERE expression supports:
      - Parentheses: ( ... )
      - Logical operators: AND, OR, NOT
      - Comparisons: =, !=, <>, <, <=, >, >=
      - Literals: numbers, quoted strings ('...' or "..."), TRUE, FALSE, NULL
      - Identifiers map to record keys

    Raises:
      ValueError if the SQL is malformed or evaluation fails.
    """
    if not isinstance(sql_command, str) or not sql_command.strip():
        raise ValueError("SQL command must be a non-empty string")

    # Parse the SQL command
    select_part, where_part, order_part = _parse_sql(sql_command)

    # Filter
    filtered = records
    if where_part is not None:
        rpn = _where_to_rpn(where_part)
        filtered = [r for r in records if _eval_rpn(rpn, r)]

    # Project (select columns)
    results: List[Dict[str, Any]]
    if select_part == '*':
        # Return shallow copies to avoid accidental mutation of originals
        results = [dict(r) for r in filtered]
    else:
        cols = [c.strip() for c in select_part.split(',') if c.strip()]
        if not cols:
            raise ValueError("SELECT clause must specify at least one column or *")
        results = select_columns(filtered, cols)

    # Order
    if order_part is not None:
        order_specs = _parse_order_by(order_part)
        # Stable multi-key sort: apply from last to first
        for col, desc in reversed(order_specs):
            def sort_key(v):
                val = v.get(col, None)
                # Normalize to ensure consistent and safe comparisons
                if val is None:
                    return (1, '')
                # If value is inherently orderable, keep; otherwise, coerce to string
                try:
                    _ = val < val  # probe comparability without altering order
                    return (0, val)
                except Exception:
                    return (0, str(val))
            results.sort(key=sort_key, reverse=desc)

    return results


# ---------- Internal helpers for SQL parsing/evaluation ----------

_SQL_REGEX = re.compile(
    r"""
    ^\s*SELECT\s+
    (?P<select>.*?)                                   # select list or *
    (?:\s+WHERE\s+(?P<where>.*?))?                    # optional WHERE (non-greedy)
    (?:\s+ORDER\s+BY\s+(?P<order>.*))?                # optional ORDER BY (rest)
    \s*$
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

def _parse_sql(sql: str):
    m = _SQL_REGEX.match(sql)
    if not m:
        raise ValueError("Malformed SQL: could not parse")
    select_part = m.group('select').strip()
    where_part = m.group('where')
    order_part = m.group('order')
    # Normalize select
    if not select_part:
        raise ValueError("Malformed SQL: SELECT clause is empty")
    if select_part.strip() == '*':
        select_part = '*'
    return select_part, (where_part.strip() if where_part else None), (order_part.strip() if order_part else None)


# Tokenization for WHERE clause
_TOKEN_RE = re.compile(
    r"""
    (?P<SPACE>\s+)
    |(?P<LPAREN>\()
    |(?P<RPAREN>\))
    |(?P<OP><=|>=|<>|!=|=|<|>)
    |(?P<LOGIC>\bAND\b|\bOR\b|\bNOT\b)
    |(?P<STRING>'([^'\\]|\\.)*'|"([^"\\]|\\.)*")
    |(?P<NUMBER>\d+\.\d+|\d+)
    |(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)
    |(?P<OTHER>.)
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

def _unescape_string(s: str) -> str:
    # Strip outer quotes and unescape common escapes
    quote = s[0]
    inner = s[1:-1]
    try:
        # Handle backslash escapes like \n, \t, \", \'
        inner = bytes(inner, 'utf-8').decode('unicode_escape')
    except Exception:
        pass
    return inner

def _tokenize_where(expr: str):
    tokens = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise ValueError(f"Malformed WHERE clause near: {expr[pos:]}")
        kind = m.lastgroup
        text = m.group(0)
        pos = m.end()
        if kind == 'SPACE':
            continue
        if kind == 'LPAREN':
            tokens.append(('LPAREN', '('))
        elif kind == 'RPAREN':
            tokens.append(('RPAREN', ')'))
        elif kind == 'OP':
            tokens.append(('OP', text.upper()))
        elif kind == 'LOGIC':
            tokens.append(('LOGIC', text.upper()))
        elif kind == 'STRING':
            tokens.append(('LIT', _unescape_string(text)))
        elif kind == 'NUMBER':
            if '.' in text:
                tokens.append(('LIT', float(text)))
            else:
                tokens.append(('LIT', int(text)))
        elif kind == 'IDENT':
            upper = text.upper()
            if upper == 'TRUE':
                tokens.append(('LIT', True))
            elif upper == 'FALSE':
                tokens.append(('LIT', False))
            elif upper == 'NULL':
                tokens.append(('LIT', None))
            else:
                tokens.append(('IDENT', text))
        elif kind == 'OTHER':
            raise ValueError(f"Unexpected character in WHERE clause: {text!r}")
    return tokens

_PRECEDENCE = {
    'NOT': 4,
    '=': 3, '!=': 3, '<>': 3, '<': 3, '<=': 3, '>': 3, '>=': 3,
    'AND': 2,
    'OR': 1,
}
_RIGHT_ASSOC = {'NOT'}

def _where_to_rpn(expr: str):
    tokens = _tokenize_where(expr)
    output = []
    ops = []

    def push_op(op):
        while ops:
            top = ops[-1]
            if top in ('(',):
                break
            if (_PRECEDENCE.get(top, 0) > _PRECEDENCE.get(op, 0)) or (
                _PRECEDENCE.get(top, 0) == _PRECEDENCE.get(op, 0) and op not in _RIGHT_ASSOC
            ):
                output.append(ops.pop())
            else:
                break
        ops.append(op)

    for kind, value in tokens:
        if kind == 'LIT':
            output.append(('LIT', value))
        elif kind == 'IDENT':
            output.append(('IDENT', value))
        elif kind == 'LOGIC':
            op = value  # AND, OR, NOT
            push_op(op)
        elif kind == 'OP':
            push_op(value)  # comparison operators
        elif kind == 'LPAREN':
            ops.append('(')
        elif kind == 'RPAREN':
            found = False
            while ops:
                top = ops.pop()
                if top == '(':
                    found = True
                    break
                output.append(top)
            if not found:
                raise ValueError("Mismatched parentheses in WHERE clause")
    while ops:
        top = ops.pop()
        if top == '(':
            raise ValueError("Mismatched parentheses in WHERE clause")
        output.append(top)
    return output

def _safe_compare(op_func, left, right) -> bool:
    try:
        return bool(op_func(left, right))
    except Exception:
        return False

_COMP_OPS = {
    '=': operator.eq,
    '!=': operator.ne,
    '<>': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
}

def _eval_rpn(rpn, record: Dict[str, Any]) -> bool:
    stack = []
    for token in rpn:
        if isinstance(token, tuple):
            kind, value = token
            if kind == 'LIT':
                stack.append(value)
            elif kind == 'IDENT':
                stack.append(record.get(value, None))
            else:
                raise ValueError(f"Invalid token in evaluation: {token}")
        else:
            op = token  # operator as string
            if op == 'NOT':
                if not stack:
                    raise ValueError("Malformed WHERE: NOT missing operand")
                val = stack.pop()
                stack.append(not bool(val))
            elif op in ('AND', 'OR'):
                if len(stack) < 2:
                    raise ValueError("Malformed WHERE: logical operator missing operands")
                b = bool(stack.pop())
                a = bool(stack.pop())
                stack.append(a and b if op == 'AND' else a or b)
            elif op in _COMP_OPS:
                if len(stack) < 2:
                    raise ValueError("Malformed WHERE: comparison missing operands")
                right = stack.pop()
                left = stack.pop()
                comp = _COMP_OPS[op]
                stack.append(_safe_compare(comp, left, right))
            else:
                raise ValueError(f"Unknown operator in WHERE: {op}")
    if len(stack) != 1:
        raise ValueError("Malformed WHERE: could not evaluate expression")
    return bool(stack[0])

def _parse_order_by(order_clause: str):
    # Split by commas not inside quotes
    parts = []
    buf = []
    in_single = False
    in_double = False
    for ch in order_clause:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == ',' and not in_single and not in_double:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append(''.join(buf).strip())

    specs = []
    for part in parts:
        if not part:
            continue
        tokens = part.split()
        if not tokens:
            continue
        col = tokens[0]
        direction = tokens[1].upper() if len(tokens) > 1 else 'ASC'
        if direction not in ('ASC', 'DESC'):
            raise ValueError(f"Invalid ORDER BY direction for {col}: {direction}")
        specs.append((col, direction == 'DESC'))
    if not specs:
        raise ValueError("ORDER BY clause is empty or malformed")
    return specs
