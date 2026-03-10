import re
import operator
from functools import partial
from typing import Iterable, Mapping, Any, List, Dict, Sequence, Optional


def extract_fields(records: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Create a new list of dictionaries containing only the specified fields from each record.

    Behavior:
    - Only includes keys that exist in each record (missing keys are skipped).
    - The order of keys in the output dictionaries follows the order provided in `fields`.
    - Duplicate field names in `fields` are de-duplicated, preserving their first occurrence.

    Args:
        records: An iterable of mapping objects (e.g., list of dicts).
        fields: A sequence of field names to include.

    Returns:
        A list of dictionaries, each containing only the specified fields present in the original records.

    Raises:
        TypeError: If any record is not a mapping/dict.
    """
    if records is None:
        return []
    if fields is None:
        return []

    # De-duplicate fields while preserving order
    seen = set()
    ordered_fields: List[str] = []
    for f in fields:
        if f not in seen:
            seen.add(f)
            ordered_fields.append(f)

    result: List[Dict[str, Any]] = []
    for rec in records:
        if rec is None:
            continue
        if not isinstance(rec, Mapping):
            raise TypeError(f"Each record must be a mapping/dict, got {type(rec).__name__}")
        filtered = {k: rec[k] for k in ordered_fields if k in rec}
        result.append(filtered)

    return result


def filter_and_extract(
    records: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    conditions: Optional[Mapping[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Filter records by equality-based conditions and extract only the specified fields.

    Args:
        records: An iterable of mapping objects (e.g., list of dicts).
        fields: A sequence of field names to include in the output.
        conditions: A mapping of field -> required value; only records where all specified
                    fields equal the given values will be included. If None or empty, no filtering is applied.

    Returns:
        A list of dictionaries corresponding to the filtered records, each containing only the requested fields.

    Raises:
        TypeError: If any record is not a mapping/dict.
    """
    if records is None:
        return []

    if not conditions:
        return extract_fields(records, fields)

    filtered_records: List[Mapping[str, Any]] = []
    sentinel = object()

    for rec in records:
        if rec is None:
            continue
        if not isinstance(rec, Mapping):
            raise TypeError(f"Each record must be a mapping/dict, got {type(rec).__name__}")
        match = True
        for k, v in conditions.items():
            if rec.get(k, sentinel) != v:
                match = False
                break
        if match:
            filtered_records.append(rec)

    return extract_fields(filtered_records, fields)


def run_sql_query(dataset: List[Mapping[str, Any]], sql_query: str) -> List[Dict[str, Any]]:
    """
    Execute a SQL-like query against an in-memory dataset (list of dictionaries).

    Supported syntax (case-insensitive keywords):
      SELECT <fields> FROM dataset
        [WHERE <expression>]
        [ORDER BY <field1> [ASC|DESC] [, <field2> [ASC|DESC] ...]]
        [LIMIT <n>]

    - <fields> is either '*' or a comma-separated list of identifiers.
    - <expression> supports:
         - Comparisons: =, !=, <>, <, <=, >, >=
         - IN (value1, value2, ...)
         - LIKE '<pattern>' where % matches any sequence and _ matches a single character
         - Boolean operators: NOT, AND, OR
         - Parentheses for grouping
         - Literals: numbers, strings in single quotes (use '' to escape a single quote),
                     TRUE, FALSE, NULL
    - If a field is missing in a record, its value is treated as None.
    - Malformed queries or execution failures raise ValueError.

    Args:
        dataset: List of mapping objects representing rows.
        sql_query: SQL-like query string.

    Returns:
        A list of dictionaries with the query results.
    """
    if not isinstance(dataset, list):
        raise ValueError("dataset must be a list of dictionaries")
    for i, rec in enumerate(dataset):
        if not isinstance(rec, Mapping):
            raise ValueError(f"dataset element at index {i} is not a mapping/dict")

    if not isinstance(sql_query, str) or not sql_query.strip():
        raise ValueError("sql_query must be a non-empty string")

    main_pat = re.compile(
        r'^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<from>\w+)'
        r'(?:\s+WHERE\s+(?P<where>.+?))?'
        r'(?:\s+ORDER\s+BY\s+(?P<order>.+?))?'
        r'(?:\s+LIMIT\s+(?P<limit>\d+))?\s*;?\s*$',
        re.IGNORECASE | re.DOTALL
    )
    m = main_pat.match(sql_query)
    if not m:
        raise ValueError("Malformed query")

    select_clause = m.group('select').strip()
    from_name = m.group('from').strip()
    where_clause = (m.group('where') or '').strip()
    order_clause = (m.group('order') or '').strip()
    limit_clause = (m.group('limit') or '').strip()

    if from_name.lower() != 'dataset':
        raise ValueError("FROM must reference 'dataset'")

    # Parse fields
    select_all = False
    fields: List[str] = []
    if select_clause == '*':
        select_all = True
    else:
        parts = [p.strip() for p in select_clause.split(',') if p.strip()]
        ident_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
        for p in parts:
            # Support simple "field" (ignore AS/alias for simplicity)
            if not ident_re.match(p):
                raise ValueError(f"Invalid field in SELECT clause: {p}")
            fields.append(p)

    # Tokenizer and parser for WHERE
    def tokenize(s: str):
        tokens = []
        pos = 0
        length = len(s)

        def match(pattern: str, flags=0):
            return re.match(pattern, s[pos:], flags)

        while pos < length:
            # Whitespace
            m_ws = match(r'\s+')
            if m_ws:
                pos += m_ws.end()
                continue

            # Parentheses and comma
            for typ, pat in (('LPAREN', r'\('), ('RPAREN', r'\)'), ('COMMA', r',')):
                m_sym = match(pat)
                if m_sym:
                    tokens.append((typ, m_sym.group(0)))
                    pos += m_sym.end()
                    break
            else:
                # Operators
                m_op = match(r'(<=|>=|!=|<>|=|<|>)')
                if m_op:
                    tokens.append(('OP', m_op.group(1)))
                    pos += m_op.end()
                    continue

                # Keywords and booleans/null
                for typ, pat in (
                    ('IN', r'(?i)\bIN\b'),
                    ('LIKE', r'(?i)\bLIKE\b'),
                    ('AND', r'(?i)\bAND\b'),
                    ('OR', r'(?i)\bOR\b'),
                    ('NOT', r'(?i)\bNOT\b'),
                    ('TRUE', r'(?i)\bTRUE\b'),
                    ('FALSE', r'(?i)\bFALSE\b'),
                    ('NULL', r'(?i)\bNULL\b'),
                ):
                    m_kw = match(pat)
                    if m_kw:
                        tokens.append((typ, m_kw.group(0)))
                        pos += m_kw.end()
                        break
                else:
                    # String literal
                    m_str = match(r"'(?:[^'\\]|\\.|'')*'")
                    if m_str:
                        tokens.append(('STRING', m_str.group(0)))
                        pos += m_str.end()
                        continue

                    # Number
                    m_num = match(r'\d+(?:\.\d+)?')
                    if m_num:
                        tokens.append(('NUMBER', m_num.group(0)))
                        pos += m_num.end()
                        continue

                    # Identifier
                    m_ident = match(r'[A-Za-z_][A-Za-z0-9_]*')
                    if m_ident:
                        tokens.append(('IDENT', m_ident.group(0)))
                        pos += m_ident.end()
                        continue

                    raise ValueError(f"Invalid token near: {s[pos:pos+20]!r}")
        return tokens

    def unescape_sql_string(lit: str) -> str:
        # Strip surrounding single quotes
        inner = lit[1:-1]
        # Unescape doubled single quotes and backslash escapes
        inner = inner.replace("''", "'")
        inner = re.sub(r'\\(.)', r'\1', inner)
        return inner

    def parse_value(tok_type: str, tok_val: str) -> Any:
        if tok_type == 'STRING':
            return unescape_sql_string(tok_val)
        if tok_type == 'NUMBER':
            return int(tok_val) if '.' not in tok_val else float(tok_val)
        if tok_type == 'TRUE':
            return True
        if tok_type == 'FALSE':
            return False
        if tok_type == 'NULL':
            return None
        raise ValueError(f"Expected a literal value, got {tok_type} {tok_val!r}")

    tokens = tokenize(where_clause) if where_clause else []

    # Recursive-descent parser with precedence: NOT > AND > OR
    idx = 0

    def peek():
        return tokens[idx] if idx < len(tokens) else None

    def consume(expected_type=None, expected_value=None):
        nonlocal idx
        tok = peek()
        if not tok:
            raise ValueError("Unexpected end of WHERE clause")
        if expected_type and tok[0] != expected_type:
            raise ValueError(f"Expected {expected_type}, got {tok[0]}")
        if expected_value and tok[1].upper() != expected_value.upper():
            raise ValueError(f"Expected {expected_value}, got {tok[1]}")
        idx += 1
        return tok

    def parse_expr():
        return parse_or()

    def parse_or():
        left = parse_and()
        while True:
            tok = peek()
            if tok and tok[0] == 'OR':
                consume('OR')
                right = parse_and()
                left = ('OR', left, right)
            else:
                break
        return left

    def parse_and():
        left = parse_not()
        while True:
            tok = peek()
            if tok and tok[0] == 'AND':
                consume('AND')
                right = parse_not()
                left = ('AND', left, right)
            else:
                break
        return left

    def parse_not():
        tok = peek()
        if tok and tok[0] == 'NOT':
            consume('NOT')
            node = parse_not()
            return ('NOT', node)
        return parse_primary()

    def parse_primary():
        tok = peek()
        if tok and tok[0] == 'LPAREN':
            consume('LPAREN')
            node = parse_expr()
            if not peek() or peek()[0] != 'RPAREN':
                raise ValueError("Missing closing parenthesis")
            consume('RPAREN')
            return node
        return parse_predicate()

    def parse_predicate():
        # IDENT <op> <value> | IDENT IN (values) | IDENT LIKE <value>
        tok = peek()
        if not tok or tok[0] not in ('IDENT',):
            raise ValueError("Expected field name in predicate")
        field = consume('IDENT')[1]

        nxt = peek()
        if not nxt:
            raise ValueError("Unexpected end after field name")

        if nxt[0] == 'IN':
            consume('IN')
            if not peek() or peek()[0] != 'LPAREN':
                raise ValueError("Expected '(' after IN")
            consume('LPAREN')
            values: List[Any] = []
            first = True
            while True:
                if not first:
                    if peek() and peek()[0] == 'COMMA':
                        consume('COMMA')
                    else:
                        break
                tokv = peek()
                if not tokv or tokv[0] not in ('STRING', 'NUMBER', 'TRUE', 'FALSE', 'NULL'):
                    if first:
                        raise ValueError("Expected at least one value in IN (...)")
                    break
                values.append(parse_value(*consume()))
                first = False
            if not peek() or peek()[0] != 'RPAREN':
                raise ValueError("Expected ')' to close IN list")
            consume('RPAREN')
            return ('IN', field, values)

        if nxt[0] == 'LIKE':
            consume('LIKE')
            tokv = consume()
            if tokv[0] not in ('STRING',):
                raise ValueError("LIKE expects a string pattern")
            val = parse_value(*tokv)
            return ('LIKE', field, val)

        if nxt[0] == 'OP':
            op = consume('OP')[1]
            tokv = consume()
            if tokv[0] not in ('STRING', 'NUMBER', 'TRUE', 'FALSE', 'NULL'):
                raise ValueError("Expected a literal after operator")
            val = parse_value(*tokv)
            return ('CMP', op, field, val)

        raise ValueError("Invalid predicate")

    ast = None
    if tokens:
        ast = parse_expr()
        if idx != len(tokens):
            raise ValueError("Unexpected tokens at end of WHERE clause")

    # Build evaluator
    cmp_ops = {
        '=': operator.eq,
        '!=': operator.ne,
        '<>': operator.ne,
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge,
    }

    def like_to_regex(pat: str) -> re.Pattern:
        # Convert SQL LIKE to regex
        # Escape regex special chars, then replace % -> .* and _ -> .
        escaped = ''
        for ch in pat:
            if ch == '%':
                escaped += '.*'
            elif ch == '_':
                escaped += '.'
            else:
                escaped += re.escape(ch)
        return re.compile('^' + escaped + '$')

    def eval_ast(node, rec: Mapping[str, Any]) -> bool:
        if node is None:
            return True
        typ = node[0]
        if typ == 'OR':
            return eval_ast(node[1], rec) or eval_ast(node[2], rec)
        if typ == 'AND':
            return eval_ast(node[1], rec) and eval_ast(node[2], rec)
        if typ == 'NOT':
            return not eval_ast(node[1], rec)
        if typ == 'IN':
            _, field, values = node
            left = rec.get(field, None)
            try:
                return left in values
            except Exception:
                return False
        if typ == 'LIKE':
            _, field, pattern = node
            left = rec.get(field, None)
            if not isinstance(left, str):
                return False
            try:
                regex = like_to_regex(pattern)
                return bool(regex.match(left))
            except Exception:
                return False
        if typ == 'CMP':
            _, op, field, right = node
            left = rec.get(field, None)
            func = cmp_ops.get(op)
            if func is None:
                return False
            try:
                return func(left, right)
            except Exception:
                return False
        raise ValueError("Invalid expression tree")

    # Apply WHERE
    filtered = [rec for rec in dataset if eval_ast(ast, rec)]

    # ORDER BY
    def norm_value(v: Any):
        # Provide a consistent ordering across types and None
        if v is None:
            return (0, 0)
        if isinstance(v, bool):
            return (1, int(v))
        if isinstance(v, (int, float)):
            return (2, float(v))
        if isinstance(v, str):
            return (3, v.lower())
        return (4, str(v))

    def parse_order_by(clause: str):
        if not clause:
            return []
        items = [c.strip() for c in clause.split(',') if c.strip()]
        order_specs = []
        for item in items:
            m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*(ASC|DESC)?$', item, flags=re.IGNORECASE)
            if not m:
                raise ValueError(f"Invalid ORDER BY item: {item}")
            field = m.group(1)
            dir_token = (m.group(2) or 'ASC').upper()
            reverse = dir_token == 'DESC'
            order_specs.append((field, reverse))
        return order_specs

    order_specs = parse_order_by(order_clause)
    # Stable sort: apply from last to first
    for field, reverse in reversed(order_specs):
        filtered.sort(key=lambda r, f=field: norm_value(r.get(f, None)), reverse=reverse)

    # LIMIT
    if limit_clause:
        try:
            limit_n = int(limit_clause)
        except Exception:
            raise ValueError("LIMIT must be an integer")
        if limit_n < 0:
            raise ValueError("LIMIT must be non-negative")
        filtered = filtered[:limit_n]

    # SELECT fields
    if select_all:
        return [dict(rec) for rec in filtered]
    else:
        return extract_fields(filtered, fields)
