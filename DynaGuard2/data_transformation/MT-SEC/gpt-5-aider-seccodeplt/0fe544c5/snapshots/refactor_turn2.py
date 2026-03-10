import re
import operator
from functools import partial
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional


class _Token:
    def __init__(self, typ: str, val: Any = None):
        self.type = typ  # e.g., IDENT, STRING, NUMBER, AND, OR, NOT, LPAREN, RPAREN, OP, LIKE, TRUE, FALSE, NULL
        self.value = val

    def __repr__(self):
        return f"_Token({self.type!r}, {self.value!r})"


def _tokenize_where(expr: str) -> List[_Token]:
    s = expr
    i = 0
    n = len(s)
    tokens: List[_Token] = []

    def peek(offset: int = 0) -> str:
        j = i + offset
        return s[j] if 0 <= j < n else ""

    while i < n:
        ch = s[i]

        # Skip whitespace
        if ch.isspace():
            i += 1
            continue

        # Parentheses
        if ch == '(':
            tokens.append(_Token('LPAREN', '('))
            i += 1
            continue
        if ch == ')':
            tokens.append(_Token('RPAREN', ')'))
            i += 1
            continue

        # Multi-char operators
        if s[i:i+2] in ('<=', '>=', '!='):
            op = s[i:i+2]
            tokens.append(_Token('OP', op))
            i += 2
            continue

        # Alternate not-equal SQL operator <>
        if s[i:i+2] == '<>':
            tokens.append(_Token('OP', '!='))
            i += 2
            continue

        # Single-char operators
        if ch in ('=', '<', '>'):
            tokens.append(_Token('OP', ch))
            i += 1
            continue

        # String literals '...' or "..."
        if ch in ("'", '"'):
            quote = ch
            i += 1
            buf = []
            escaped = False
            while i < n:
                c = s[i]
                if escaped:
                    # simple escape handling
                    escapes = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}
                    buf.append(escapes.get(c, c))
                    escaped = False
                    i += 1
                else:
                    if c == '\\':
                        escaped = True
                        i += 1
                    elif c == quote:
                        i += 1
                        break
                    else:
                        buf.append(c)
                        i += 1
            else:
                raise ValueError("Unterminated string literal in WHERE clause")
            tokens.append(_Token('STRING', ''.join(buf)))
            continue

        # Numbers: optional sign, digits, optional decimal/exponent
        num_match = re.match(r'[+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+)', s[i:])
        if num_match:
            txt = num_match.group(0)
            i += len(txt)
            # Try int if possible
            try:
                if re.match(r'^[+-]?\d+$', txt):
                    val = int(txt)
                else:
                    val = float(txt)
            except ValueError:
                val = txt
            tokens.append(_Token('NUMBER', val))
            continue

        # Identifiers and keywords
        ident_match = re.match(r'[A-Za-z_][A-Za-z0-9_\.]*', s[i:])
        if ident_match:
            ident = ident_match.group(0)
            i += len(ident)
            upper = ident.upper()
            if upper == 'AND':
                tokens.append(_Token('AND'))
            elif upper == 'OR':
                tokens.append(_Token('OR'))
            elif upper == 'NOT':
                tokens.append(_Token('NOT'))
            elif upper == 'LIKE':
                tokens.append(_Token('LIKE'))
            elif upper == 'TRUE':
                tokens.append(_Token('BOOLEAN', True))
            elif upper == 'FALSE':
                tokens.append(_Token('BOOLEAN', False))
            elif upper == 'NULL' or upper == 'NONE':
                tokens.append(_Token('NULL', None))
            else:
                tokens.append(_Token('IDENT', ident))
            continue

        raise ValueError(f"Unexpected character in WHERE clause: '{ch}' at position {i}")

    return tokens


class _Parser:
    def __init__(self, tokens: List[_Token]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Optional[_Token]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _pop(self) -> _Token:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of WHERE clause")
        self.pos += 1
        return tok

    def _accept(self, typ: str, values: Optional[Iterable[Any]] = None) -> Optional[_Token]:
        tok = self._peek()
        if tok and tok.type == typ and (values is None or tok.value in values):
            self.pos += 1
            return tok
        return None

    def parse(self) -> Callable[[Dict[str, Any]], bool]:
        expr = self._parse_or()
        if self._peek() is not None:
            raise ValueError("Unexpected tokens after WHERE expression")
        # Ensure boolean result
        def predicate(rec: Dict[str, Any]) -> bool:
            try:
                return bool(expr(rec))
            except Exception as e:
                raise ValueError(f"Error evaluating WHERE expression: {e}")
        return predicate

    def _parse_or(self):
        left = self._parse_and()
        while True:
            if self._accept('OR'):
                right = self._parse_and()
                left = (lambda l=left, r=right: (lambda rec: bool(l(rec)) or bool(r(rec))))
            else:
                break
        return left

    def _parse_and(self):
        left = self._parse_not()
        while True:
            if self._accept('AND'):
                right = self._parse_not()
                left = (lambda l=left, r=right: (lambda rec: bool(l(rec)) and bool(r(rec))))
            else:
                break
        return left

    def _parse_not(self):
        if self._accept('NOT'):
            operand = self._parse_not()
            return (lambda op=operand: (lambda rec: not bool(op(rec))))
        return self._parse_comparison()

    def _parse_comparison(self):
        left = self._parse_primary()

        tok = self._peek()
        if tok and (tok.type == 'OP' or tok.type == 'LIKE'):
            op_token = self._pop()
            right = self._parse_primary()
            if op_token.type == 'LIKE':
                return lambda rec, l=left, r=right: _apply_like(l(rec), r(rec))
            else:
                return lambda rec, l=left, r=right, op=op_token.value: _apply_comparison(op, l(rec), r(rec))
        # No operator -> interpret primary as boolean truthiness
        return left

    def _parse_primary(self):
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of expression")

        if self._accept('LPAREN'):
            inner = self._parse_or()
            if not self._accept('RPAREN'):
                raise ValueError("Missing closing parenthesis")
            return inner

        tok = self._peek()
        if tok.type == 'IDENT':
            ident = self._pop().value
            return lambda rec, k=ident: _get_value(k, rec)
        if tok.type == 'STRING':
            val = self._pop().value
            return lambda _rec, v=val: v
        if tok.type == 'NUMBER':
            val = self._pop().value
            return lambda _rec, v=val: v
        if tok.type == 'BOOLEAN':
            val = self._pop().value
            return lambda _rec, v=val: v
        if tok.type == 'NULL':
            self._pop()
            return lambda _rec: None

        raise ValueError(f"Unexpected token {tok}")


def _get_value(key: str, record: Dict[str, Any]) -> Any:
    # Support dotted keys for nested dicts: a.b.c
    if '.' in key:
        cur: Any = record
        for part in key.split('.'):
            if isinstance(cur, dict) and part in cur:
                cur = cur.get(part)
            else:
                return None
        return cur
    return record.get(key)


def _to_number(val: Any) -> Optional[float]:
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.strip())
        except Exception:
            return None
    return None


def _apply_comparison(op: str, left: Any, right: Any) -> bool:
    # Handle None explicitly (SQL uses IS/IS NOT for NULL; with = comparisons, we mimic common expectations)
    if left is None or right is None:
        if op in ('=', '=='):
            return left is None and right is None
        if op == '!=':
            return (left is None) != (right is None)
        # For order comparisons with NULLs, return False
        return False

    # Normalize operator
    if op == '=':
        op = '=='

    # Try numeric comparison if both numeric-like
    ln = _to_number(left)
    rn = _to_number(right)
    if ln is not None and rn is not None:
        comp_ops = {
            '==': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '<=': operator.le,
            '>': operator.gt,
            '>=': operator.ge,
        }
        if op not in comp_ops:
            raise ValueError(f"Unsupported operator: {op}")
        return bool(comp_ops[op](ln, rn))

    # Fallback to string comparison
    lstr = str(left)
    rstr = str(right)

    comp_ops = {
        '==': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge,
    }

    if op not in comp_ops:
        raise ValueError(f"Unsupported operator: {op}")
    return bool(comp_ops[op](lstr, rstr))


def _like_to_regex(pattern: str) -> re.Pattern:
    # Convert SQL LIKE with % (any length) and _ (single char) to regex
    # Escape regex special chars first, then replace
    escaped = []
    for ch in pattern:
        if ch == '%':
            escaped.append('.*')
        elif ch == '_':
            escaped.append('.')
        else:
            escaped.append(re.escape(ch))
    regex = '^' + ''.join(escaped) + '$'
    return re.compile(regex, flags=re.IGNORECASE)


def _apply_like(value: Any, pattern: Any) -> bool:
    if value is None or pattern is None:
        return False
    rx = _like_to_regex(str(pattern))
    return bool(rx.match(str(value)))


def _parse_select_list(select_part: str) -> List[Tuple[str, str]]:
    """
    Returns list of (source_field, output_key). If select is '*', returns [('*', '*')].
    Supports: column, column AS alias
    """
    if not select_part or not select_part.strip():
        raise ValueError("SELECT clause is empty")

    if select_part.strip() == '*':
        return [('*', '*')]

    items = [seg.strip() for seg in select_part.split(',')]
    result: List[Tuple[str, str]] = []
    for item in items:
        # Support optional AS alias
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:AS\s+([A-Za-z_][A-Za-z0-9_]*))?$', item, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Invalid SELECT item: '{item}' (only simple identifiers and optional AS alias supported)")
        src = m.group(1)
        alias = m.group(2) or src.split('.')[-1]
        result.append((src, alias))
    return result


def _parse_order_by(order_part: str) -> List[Tuple[str, bool]]:
    """
    Returns list of (field, ascending: bool) pairs.
    """
    if not order_part or not order_part.strip():
        raise ValueError("ORDER BY clause is empty")
    specs: List[Tuple[str, bool]] = []
    for seg in order_part.split(','):
        seg = seg.strip()
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_\.]*)\s*(ASC|DESC)?$', seg, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Invalid ORDER BY segment: '{seg}'")
        field = m.group(1)
        direction = m.group(2).upper() if m.group(2) else 'ASC'
        specs.append((field, direction == 'ASC'))
    return specs


def _normalize_sort_value(val: Any) -> Tuple[int, Any]:
    """
    Normalize value for sorting to avoid TypeError between incomparable types.
    Returns a tuple (null_flag, normalized_value) so None values sort after non-None for ascending.
    When descending, Python's reverse=True will invert ordering including null_flag.
    """
    if val is None:
        return (1, None)
    # Try numeric
    num = _to_number(val)
    if num is not None:
        return (0, num)
    # Fallback to case-insensitive string
    return (0, str(val).lower())


def _sort_records(records: List[Dict[str, Any]], order_specs: List[Tuple[str, bool]]) -> List[Dict[str, Any]]:
    # Perform stable sorts from last key to first
    out = list(records)
    for field, ascending in reversed(order_specs):
        out.sort(key=lambda rec, f=field: _normalize_sort_value(_get_value(f, rec)), reverse=not ascending)
    return out


@dataclass(frozen=True)
class QueryPlan:
    select_list: List[Tuple[str, str]]
    order_specs: List[Tuple[str, bool]]
    predicate: Callable[[Dict[str, Any]], bool]


def parse_sql_query(sql_command: str) -> QueryPlan:
    """
    Pure function: parse the SQL-like string into a query plan (select list, order specs, predicate).
    Raises ValueError on invalid input.
    """
    if not isinstance(sql_command, str) or not sql_command.strip():
        raise ValueError("sql_command must be a non-empty string")

    query = sql_command.strip()
    m = re.match(
        r'^\s*SELECT\s+(?P<select>.+?)\s*'
        r'(?:FROM\s+(?P<from>.+?)\s*)?'
        r'(?:WHERE\s+(?P<where>.+?)\s*)?'
        r'(?:ORDER\s+BY\s+(?P<order_by>.+?)\s*)?$',
        query,
        flags=re.IGNORECASE | re.DOTALL
    )
    if not m:
        raise ValueError("Invalid query. Expected syntax: SELECT <columns> [FROM <...>] [WHERE <...>] [ORDER BY <...>]")

    select_part = (m.group('select') or '').strip()
    where_part = (m.group('where') or '').strip()
    order_part = (m.group('order_by') or '').strip()

    select_list = _parse_select_list(select_part)

    # Build predicate
    if where_part:
        tokens = _tokenize_where(where_part)
        parser = _Parser(tokens)
        predicate = parser.parse()
    else:
        predicate = lambda _rec: True

    # Build order specs
    order_specs = _parse_order_by(order_part) if order_part else []

    return QueryPlan(select_list=select_list, order_specs=order_specs, predicate=predicate)


def execute_query(records: List[Dict[str, Any]], plan: QueryPlan) -> List[Dict[str, Any]]:
    """
    Pure function: execute a parsed query plan against the given records.
    Does not mutate input records. Raises ValueError on invalid input.
    """
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("records must be a list of dictionaries")

    # Filter
    filtered = [rec for rec in records if plan.predicate(rec)]

    # Order
    if plan.order_specs:
        filtered = _sort_records(filtered, plan.order_specs)

    # Projection
    result: List[Dict[str, Any]] = []
    if len(plan.select_list) == 1 and plan.select_list[0] == ('*', '*'):
        for rec in filtered:
            result.append(dict(rec))
    else:
        for rec in filtered:
            out: Dict[str, Any] = {}
            for src, alias in plan.select_list:
                out[alias] = _get_value(src, rec)
            result.append(out)

    return result


def handle_sql_query(records: List[Dict[str, Any]], sql_command: str) -> List[Dict[str, Any]]:
    """
    Execute a simple SQL-like query over a list of dictionaries.

    Supported syntax (case-insensitive keywords):
      SELECT <columns> [FROM <ignored>] [WHERE <expression>] [ORDER BY <field> [ASC|DESC], ...]
    - <columns>: '*' or comma-separated list of identifiers with optional 'AS alias'
    - WHERE: supports comparisons with =, !=, <, <=, >, >=, LIKE; logical AND, OR, NOT; parentheses; TRUE/FALSE/NULL literals
    - ORDER BY: one or more fields with optional ASC/DESC

    Returns list of dictionaries representing the result set.

    Raises ValueError on parse or execution errors.
    """
    try:
        plan = parse_sql_query(sql_command)
        return execute_query(records, plan)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to execute query: {e}")
