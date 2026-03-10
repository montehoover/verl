import re
import operator
from functools import partial

_OPS = {
    '=': operator.eq,
    '==': operator.eq,
    '!=': operator.ne,
    '<>': operator.ne,
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
}

_WHERE_TOKEN_RE = re.compile(
    r"""
    \s*(
        (?P<number>-?\d+(?:\.\d+)?)        |  # number (int or float)
        (?P<string>'[^']*'|"[^"]*")        |  # quoted string
        (?P<op>>=|<=|!=|<>|=|>|<)          |  # comparison operator
        (?P<logical>\bAND\b|\bOR\b)        |  # logical operator
        (?P<ident>[A-Za-z_][A-Za-z0-9_]*)  |  # identifier
        (?P<lparen>\()                     |  # left parenthesis (unsupported)
        (?P<rparen>\))                        # right parenthesis (unsupported)
    )\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

_QUERY_RE = re.compile(
    r"""
    ^\s*
    SELECT\s+(?P<select>.+?)\s+
    FROM\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)
    (?:\s+WHERE\s+(?P<where>.+?)(?=\s+ORDER\s+BY\s+|$))?
    (?:\s+ORDER\s+BY\s+(?P<order>.+))?
    \s*$
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def _parse_select(select_part: str):
    text = select_part.strip()
    if text == '*':
        return ['*']
    fields = [f.strip() for f in text.split(',')]
    if not all(fields) or any(' ' in f for f in fields):
        # Disallow aliases or spaces in identifiers for simplicity
        raise ValueError("Invalid SELECT field list")
    return fields


def _tokenize_where(where_part: str):
    tokens = []
    pos = 0
    while pos < len(where_part):
        m = _WHERE_TOKEN_RE.match(where_part, pos)
        if not m:
            snippet = where_part[pos:pos + 20]
            raise ValueError(f"Invalid token in WHERE clause near: {snippet!r}")
        pos = m.end()
        kind = None
        value = None
        for k in ('number', 'string', 'op', 'logical', 'ident', 'lparen', 'rparen'):
            v = m.group(k)
            if v is not None:
                kind = k
                value = v
                break
        if kind in ('lparen', 'rparen'):
            raise ValueError("Parentheses are not supported in WHERE clause")
        tokens.append((kind, value))
    return tokens


def _parse_value_token(tok):
    kind, value = tok
    if kind == 'number':
        return float(value) if ('.' in value) else int(value)
    if kind == 'string':
        # Strip surrounding quotes; no escape handling for simplicity.
        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            return value[1:-1]
        raise ValueError("Malformed quoted string in WHERE clause")
    if kind == 'ident':
        # Ident on RHS is treated as a field reference (compare two columns)
        return tok  # defer to getter as field reference
    raise ValueError("Expected a value in WHERE clause")


def _get_field(record, key: str):
    if key in record:
        return record[key]
    raise ValueError(f"Field {key!r} not found in record")


def _compile_condition(left_ident_tok, op_tok, right_tok):
    l_kind, l_name = left_ident_tok
    if l_kind != 'ident':
        raise ValueError("Left-hand side of comparison must be a field name")
    op_symbol = op_tok[1]
    op = _OPS.get(op_symbol)
    if op is None:
        raise ValueError(f"Unsupported operator {op_symbol!r} in WHERE clause")

    # Right side: either constant value or field reference
    if right_tok[0] in ('number', 'string'):
        right_value = _parse_value_token(right_tok)
        def right_getter(rec):
            return right_value
    elif right_tok[0] == 'ident':
        r_field = right_tok[1]
        def right_getter(rec):
            return _get_field(rec, r_field)
    else:
        raise ValueError("Invalid right-hand side in WHERE clause")

    def predicate(rec):
        try:
            left_val = _get_field(rec, l_name)
            right_val = right_getter(rec)
            return op(left_val, right_val)
        except ValueError:
            # propagate missing field errors
            raise
        except Exception as e:
            raise ValueError(f"Failed to evaluate WHERE condition: {e}") from e

    return predicate


def _build_where_predicate(where_part: str):
    if not where_part or not where_part.strip():
        return lambda rec: True

    tokens = _tokenize_where(where_part)

    # Build predicates with AND precedence over OR: (a AND b) OR (c AND d) ...
    i = 0
    groups = []  # list of predicates combined by OR

    def parse_condition_at(idx):
        if idx + 2 >= len(tokens):
            raise ValueError("Incomplete condition in WHERE clause")
        left_tok = tokens[idx]
        op_tok = tokens[idx + 1]
        right_tok = tokens[idx + 2]
        pred = _compile_condition(left_tok, op_tok, right_tok)
        return pred, idx + 3

    while i < len(tokens):
        # One AND-group
        if tokens[i][0] == 'logical' and tokens[i][1].upper() in ('AND', 'OR'):
            raise ValueError("Unexpected logical operator position in WHERE clause")

        pred, i = parse_condition_at(i)

        # Chain ANDs
        while i < len(tokens) and tokens[i][0] == 'logical' and tokens[i][1].upper() == 'AND':
            i += 1
            rhs_pred, i = parse_condition_at(i)
            prev_pred = pred
            pred = lambda rec, a=prev_pred, b=rhs_pred: a(rec) and b(rec)

        groups.append(pred)

        if i < len(tokens):
            if tokens[i][0] == 'logical' and tokens[i][1].upper() == 'OR':
                i += 1
                continue
            else:
                raise ValueError("Unexpected token in WHERE clause")

    if not groups:
        return lambda rec: True

    if len(groups) == 1:
        return groups[0]

    def final_pred(rec):
        for g in groups:
            if g(rec):
                return True
        return False

    return final_pred


def _parse_order_by(order_part: str):
    if not order_part or not order_part.strip():
        return []
    pieces = [p.strip() for p in order_part.split(',') if p.strip()]
    orderings = []
    for p in pieces:
        bits = p.split()
        if len(bits) == 1:
            field, direction = bits[0], 'ASC'
        elif len(bits) == 2:
            field, direction = bits
        else:
            raise ValueError("Invalid ORDER BY clause")
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', field):
            raise ValueError(f"Invalid ORDER BY field {field!r}")
        dir_upper = direction.upper()
        if dir_upper not in ('ASC', 'DESC'):
            raise ValueError("ORDER BY direction must be ASC or DESC")
        orderings.append((field, dir_upper == 'DESC'))
    return orderings


def run_sql_query(records, command):
    """
    Execute a basic SQL-like statement on a list of dictionaries.

    Supported syntax (case-insensitive):
      SELECT <fields>|* FROM <identifier> [WHERE <conditions>] [ORDER BY <field> [ASC|DESC] [, ...]]

    WHERE:
      - Comparisons: =, !=, <>, >, >=, <, <=
      - Logical: AND, OR (AND has higher precedence; no parentheses support)
      - Right-hand values may be numbers, quoted strings, or other fields (unquoted identifiers)

    ORDER BY:
      - One or more fields, each optionally followed by ASC or DESC
      - Sorting happens before projection so ordering can use non-selected fields

    Args:
        records: list of dicts
        command: SQL-like string

    Returns:
        List[dict]: query results

    Raises:
        ValueError: for malformed queries or evaluation issues
    """
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("records must be a list of dictionaries")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")

    m = _QUERY_RE.match(command)
    if not m:
        raise ValueError("Invalid query format")

    select_part = m.group('select')
    table_part = m.group('table')  # parsed but not used; dataset is provided via 'records'
    where_part = m.group('where') or ''
    order_part = m.group('order') or ''

    # Parse clauses
    select_fields = _parse_select(select_part)
    predicate = _build_where_predicate(where_part)
    orderings = _parse_order_by(order_part)

    # Filter
    try:
        filtered = [rec for rec in records if predicate(rec)]
    except ValueError:
        # propagate missing field errors etc.
        raise

    # Order (stable multi-key sort from last to first)
    if orderings:
        try:
            for field, reverse in reversed(orderings):
                filtered.sort(key=lambda r, f=field: _get_field(r, f), reverse=reverse)
        except ValueError:
            raise
        except TypeError as e:
            raise ValueError(f"Failed to sort records: {e}") from e

    # Project
    if select_fields == ['*']:
        # return shallow copies to avoid accidental external mutation
        return [dict(rec) for rec in filtered]

    result = []
    for rec in filtered:
        try:
            projected = {f: _get_field(rec, f) for f in select_fields}
        except ValueError:
            raise
        result.append(projected)

    return result
