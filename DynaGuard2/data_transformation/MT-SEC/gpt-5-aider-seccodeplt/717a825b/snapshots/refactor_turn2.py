import re
import operator
from functools import partial

# Supported comparison operators
_COMPARATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
}

_LOGICAL_AND = 'AND'
_LOGICAL_OR = 'OR'

_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _strip_quotes(value: str) -> str:
    if not value:
        return value
    if len(value) >= 2 and ((value[0] == value[-1] == "'") or (value[0] == value[-1] == '"')):
        return value[1:-1]
    return value


def _unescape_string(s: str) -> str:
    # Handle common escape sequences
    escapes = {
        r"\\": "\\",
        r"\'": "'",
        r"\"": '"',
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
        r"\b": "\b",
        r"\f": "\f",
        r"\0": "\0",
    }
    # Replace known escapes
    def repl(m):
        seq = m.group(0)
        return escapes.get(seq, seq[1])  # default to escaped char without backslash
    return re.sub(r'\\.|.', repl, s) if '\\' in s else s


def _parse_literal(token: str):
    # Booleans and NULL (case-insensitive)
    if isinstance(token, str):
        low = token.lower()
        if low == 'true':
            return True
        if low == 'false':
            return False
        if low == 'null':
            return None

    # Quoted strings
    if len(token) >= 2 and ((token[0] == token[-1] == "'") or (token[0] == token[-1] == '"')):
        inner = token[1:-1]
        return _unescape_string(inner)

    # Numbers: int or float (with optional exponent)
    num_match = re.fullmatch(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', token)
    if num_match:
        if any(c in token for c in ('.', 'e', 'E')):
            try:
                return float(token)
            except ValueError:
                pass
        else:
            try:
                return int(token)
            except ValueError:
                pass

    # Fallback to raw string without quotes
    return token


def _tokenize_where(expr: str):
    if expr is None:
        return []

    # Token specs (order matters)
    token_specification = [
        ('SKIP',       r'[ \t\r\n]+'),
        ('LPAREN',     r'\('),
        ('RPAREN',     r'\)'),
        ('OP',         r'<=|>=|!=|=|<|>'),
        ('AND',        r'(?i:\bAND\b)'),
        ('OR',         r'(?i:\bOR\b)'),
        ('SSTRING',    r"'(?:\\.|[^'])*'"),
        ('DSTRING',    r'"(?:\\.|[^"])*"'),
        ('NUMBER',     r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'),
        ('IDENT',      r'[A-Za-z_][A-Za-z0-9_]*'),
        ('MISMATCH',   r'.'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    get_token = re.compile(tok_regex).match
    pos = 0
    tokens = []
    m = get_token(expr, pos)
    while m is not None:
        kind = m.lastgroup
        value = m.group()
        if kind == 'SKIP':
            pass
        elif kind == 'LPAREN' or kind == 'RPAREN':
            tokens.append((kind, value))
        elif kind == 'OP':
            tokens.append(('OP', value))
        elif kind == 'AND':
            tokens.append(('LOGIC', _LOGICAL_AND))
        elif kind == 'OR':
            tokens.append(('LOGIC', _LOGICAL_OR))
        elif kind == 'SSTRING' or kind == 'DSTRING':
            tokens.append(('LITERAL', value))
        elif kind == 'NUMBER':
            tokens.append(('LITERAL', value))
        elif kind == 'IDENT':
            # Could be identifiers or special literals TRUE/FALSE/NULL
            low = value.lower()
            if low in ('true', 'false', 'null'):
                tokens.append(('LITERAL', value))
            else:
                tokens.append(('IDENT', value))
        elif kind == 'MISMATCH':
            raise ValueError(f"Invalid token in WHERE clause at position {m.start()}: {value!r}")
        pos = m.end()
        m = get_token(expr, pos)
    if pos != len(expr):
        raise ValueError(f"Unexpected text in WHERE clause at position {pos}")
    return tokens


def _to_rpn(tokens):
    # Shunting-yard algorithm
    out_queue = []
    op_stack = []

    # Precedence: comparisons > AND > OR
    precedence = {
        'OP': 3,       # comparison operators
        _LOGICAL_AND: 2,
        _LOGICAL_OR: 1,
    }

    for kind, value in tokens:
        if kind in ('IDENT', 'LITERAL'):
            out_queue.append((kind, value))
        elif kind == 'OP':
            while op_stack and op_stack[-1][0] in ('OP', 'LOGIC') and (
                (op_stack[-1][0] == 'OP' and precedence['OP'] >= precedence['OP']) or
                (op_stack[-1][0] == 'LOGIC' and precedence[op_stack[-1][1]] >= precedence['OP'])
            ):
                out_queue.append(op_stack.pop())
            op_stack.append(('OP', value))
        elif kind == 'LOGIC':
            while op_stack and op_stack[-1][0] in ('OP', 'LOGIC') and (
                (op_stack[-1][0] == 'OP' and precedence['OP'] >= precedence[value]) or
                (op_stack[-1][0] == 'LOGIC' and precedence[op_stack[-1][1]] >= precedence[value])
            ):
                out_queue.append(op_stack.pop())
            op_stack.append(('LOGIC', value))
        elif kind == 'LPAREN':
            op_stack.append((kind, value))
        elif kind == 'RPAREN':
            while op_stack and op_stack[-1][0] != 'LPAREN':
                out_queue.append(op_stack.pop())
            if not op_stack or op_stack[-1][0] != 'LPAREN':
                raise ValueError("Mismatched parentheses in WHERE clause")
            op_stack.pop()  # Remove LPAREN
        else:
            raise ValueError(f"Unexpected token in WHERE clause: {kind} {value}")

    while op_stack:
        if op_stack[-1][0] in ('LPAREN', 'RPAREN'):
            raise ValueError("Mismatched parentheses in WHERE clause")
        out_queue.append(op_stack.pop())

    return out_queue


def _coerce_for_compare(a, b):
    # Try to coerce both to numbers if possible
    def to_number(x):
        if isinstance(x, (int, float)):
            return x
        if isinstance(x, str):
            m = re.fullmatch(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', x)
            if m:
                try:
                    return float(x) if any(c in x for c in ('.', 'e', 'E')) else int(x)
                except ValueError:
                    return x
        return x

    a_num = to_number(a)
    b_num = to_number(b)
    if isinstance(a_num, (int, float)) and isinstance(b_num, (int, float)):
        return a_num, b_num

    # Otherwise, compare as-is
    return a, b


def _build_predicate(rpn):
    if not rpn:
        return lambda row: True

    def ident_getter(name):
        return lambda row: row.get(name, None)

    def lit_getter(value):
        return lambda row: value

    stack = []
    for kind, value in rpn:
        if kind == 'IDENT':
            stack.append(ident_getter(value))
        elif kind == 'LITERAL':
            stack.append(lit_getter(_parse_literal(value)))
        elif kind == 'OP':
            if value not in _COMPARATORS:
                raise ValueError(f"Unsupported operator: {value}")
            try:
                right = stack.pop()
                left = stack.pop()
            except IndexError:
                raise ValueError("Invalid WHERE expression: not enough operands for comparison")
            cmp_fn = _COMPARATORS[value]

            def make_cmp(lf, rf, cmpfn):
                def func(row):
                    lv = lf(row)
                    rv = rf(row)
                    # If missing identifiers, predicate evaluates to False when comparison is not equality check
                    a, b = _coerce_for_compare(lv, rv)
                    try:
                        return cmpfn(a, b)
                    except Exception:
                        # Any comparison failure -> False
                        return False
                return func

            stack.append(make_cmp(left, right, cmp_fn))
        elif kind == 'LOGIC':
            try:
                right = stack.pop()
                left = stack.pop()
            except IndexError:
                raise ValueError("Invalid WHERE expression: not enough operands for logical operator")
            if value == _LOGICAL_AND:
                stack.append(lambda row, lf=left, rf=right: bool(lf(row)) and bool(rf(row)))
            elif value == _LOGICAL_OR:
                stack.append(lambda row, lf=left, rf=right: bool(lf(row)) or bool(rf(row)))
            else:
                raise ValueError(f"Unsupported logical operator: {value}")
        else:
            raise ValueError(f"Invalid token in RPN: {kind}")

    if len(stack) != 1:
        raise ValueError("Invalid WHERE expression")

    return stack[0]


def _split_query_parts(query: str):
    # Extract SELECT, optional WHERE, optional ORDER BY using a regex that avoids greediness
    # Note: This is a simplified parser and may not handle keywords within quoted strings.
    pattern = re.compile(
        r'^\s*SELECT\s+(?P<select>.+?)\s*'
        r'(?:WHERE\s+(?P<where>.+?))?\s*'
        r'(?:ORDER\s+BY\s+(?P<order_by>[A-Za-z_][A-Za-z0-9_]*)'
        r'(?:\s+(?P<order_dir>ASC|DESC))?\s*)?$', re.IGNORECASE | re.DOTALL
    )
    m = pattern.match(query)
    if not m:
        raise ValueError("Invalid query syntax")
    select_part = m.group('select')
    where_part = m.group('where')
    order_by = m.group('order_by')
    order_dir = m.group('order_dir')
    return select_part, where_part, order_by, (order_dir.upper() if order_dir else None)


def _parse_select_fields(select_part: str):
    sp = select_part.strip()
    if sp == '*':
        return '*'
    fields = [f.strip() for f in sp.split(',')]
    if not fields or any(not f or not _IDENTIFIER_RE.match(f) for f in fields):
        raise ValueError("Invalid field list in SELECT clause")
    return fields


def _normalize_sort_value(v):
    if v is None:
        return (1, None)
    # Try numeric first
    if isinstance(v, (int, float)):
        return (0, (0, float(v)))
    if isinstance(v, str):
        # numeric string?
        m = re.fullmatch(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?', v)
        if m:
            try:
                num = float(v) if any(c in v for c in ('.', 'e', 'E')) else int(v)
                return (0, (0, float(num)))
            except ValueError:
                pass
        # lowercase for consistent ordering
        return (0, (1, v.lower()))
    # Fallback: string representation
    return (0, (2, str(v)))


def parse_query(query: str):
    """
    Pure function: parse a SQL-like query into an intermediate representation (IR).
    Returns a dict with keys: fields, where_rpn, order_by, order_dir.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    select_part, where_part, order_by, order_dir = _split_query_parts(query)
    fields = _parse_select_fields(select_part)
    tokens = _tokenize_where(where_part) if where_part else []
    rpn = _to_rpn(tokens) if tokens else []
    return {
        'fields': fields,
        'where_rpn': rpn,
        'order_by': order_by,
        'order_dir': order_dir,
    }


def _run_pipeline(data, steps):
    result = data
    for step in steps:
        result = step(result)
    return result


def execute_parsed_query(data, plan):
    """
    Pure function: execute a parsed query plan against the provided data.
    """
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")

    # Build steps
    steps = []

    # Filter step
    try:
        predicate = _build_predicate(plan.get('where_rpn') or [])
    except ValueError as e:
        raise
    def filter_step(rows):
        try:
            return [row for row in rows if isinstance(row, dict) and predicate(row)]
        except Exception as e:
            raise ValueError(f"Failed to evaluate WHERE clause: {e}")
    steps.append(filter_step)

    # Order step
    order_by = plan.get('order_by')
    order_dir = plan.get('order_dir')
    if order_by:
        if not _IDENTIFIER_RE.match(order_by):
            raise ValueError("Invalid field name in ORDER BY clause")
        reverse = (order_dir == 'DESC')
        def order_step(rows):
            try:
                return sorted(rows, key=lambda r: _normalize_sort_value(r.get(order_by, None)), reverse=reverse)
            except Exception as e:
                raise ValueError(f"Failed to sort data: {e}")
        steps.append(order_step)

    # Project step
    fields = plan.get('fields')
    def project_step(rows):
        result = []
        if fields == '*':
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Each row in data must be a dictionary")
                result.append(dict(row))
        else:
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Each row in data must be a dictionary")
                projected = {f: row.get(f, None) for f in fields}
                result.append(projected)
        return result
    steps.append(project_step)

    return _run_pipeline(data, steps)


def execute_custom_query(data, query):
    """
    Execute a simple SQL-like query against a list of dictionaries.

    Supported syntax (case-insensitive keywords):
      SELECT field1, field2 | SELECT *
      [WHERE <expr>]
      [ORDER BY field [ASC|DESC]]

    WHERE expression supports:
      - comparison operators: =, !=, <, <=, >, >=
      - logical operators: AND, OR
      - parentheses: ( )
      - literals: numbers, strings ('...' or "..."), TRUE/FALSE/NULL

    Args:
        data: list of dictionaries
        query: SQL-like query string

    Returns:
        List of dictionaries representing the query results.

    Raises:
        ValueError: for invalid query or execution issues.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    if not isinstance(data, list):
        raise ValueError("Data must be a list of dictionaries")

    plan = parse_query(query)
    return execute_parsed_query(data, plan)
