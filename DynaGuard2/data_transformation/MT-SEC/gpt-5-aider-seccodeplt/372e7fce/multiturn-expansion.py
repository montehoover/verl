import re
import operator
from functools import partial

from collections import defaultdict
from typing import Iterable, Mapping, Sequence, Dict, Any, List
from typing import Callable, List, Dict

def select_fields(records: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Return a list of dictionaries containing only the specified fields for each record.

    - records: iterable of dictionaries (data records)
    - fields: list/sequence of field names (strings) to select

    Missing fields in a record are omitted from that record's result.
    Non-dict records are ignored.
    """
    result: List[Dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        selected = {field: record[field] for field in fields if field in record}
        result.append(selected)
    return result

def filter_data(records: List[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Return a list of records that satisfy the provided condition.

    - records: list of dictionaries (data records)
    - condition: callable that accepts a record and returns True if it should be included

    Non-dict items in records are ignored.
    """
    if condition is None or not callable(condition):
        raise TypeError("condition must be a callable taking a record and returning bool")

    filtered: List[Dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if condition(record):
            filtered.append(record)
    return filtered


# --- SQL-like query execution ---

# Tokenizer for WHERE clause
_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (?P<number>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?) |
        (?P<string>'([^'\\]|\\.)*'|"([^"\\]|\\.)*") |
        (?P<op><=|>=|<>|!=|=|<|>) |
        (?P<lparen>\() |
        (?P<rparen>\)) |
        (?P<comma>,) |
        (?P<word>[A-Za-z_][A-Za-z0-9_]*)
    )
    """,
    re.VERBOSE,
)

def _unescape_string(s: str) -> str:
    if not s:
        return s
    quote = s[0]
    if quote not in ("'", '"') or s[-1] != quote:
        raise ValueError("Invalid string literal")
    body = s[1:-1]

    # Handle common escapes
    escapes = {
        r"\\": "\\",
        r"\'": "'",
        r'\"': '"',
        r"\n": "\n",
        r"\r": "\r",
        r"\t": "\t",
        r"\b": "\b",
        r"\f": "\f",
        r"\v": "\v",
        r"\0": "\0",
    }
    # Replace escapes
    i = 0
    out = []
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            seq = body[i : i + 2]
            if seq in escapes:
                out.append(escapes[seq])
                i += 2
                continue
        out.append(body[i])
        i += 1
    return "".join(out)

def _tokenize_where(where_str: str):
    tokens = []
    pos = 0
    length = len(where_str)
    while pos < length:
        m = _TOKEN_RE.match(where_str, pos)
        if not m:
            raise ValueError(f"Unexpected token near: {where_str[pos:pos+20]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "number":
            tokens.append(("NUMBER", m.group(kind), m.group(kind)))
        elif kind == "string":
            tokens.append(("STRING", m.group(kind), m.group(kind)))
        elif kind == "op":
            tokens.append(("OP", m.group(kind), m.group(kind)))
        elif kind == "lparen":
            tokens.append(("LPAREN", "(", "("))
        elif kind == "rparen":
            tokens.append(("RPAREN", ")", ")"))
        elif kind == "comma":
            tokens.append(("COMMA", ",", ","))
        elif kind == "word":
            word = m.group(kind)
            tokens.append(("WORD", word.upper(), word))
        else:
            raise ValueError("Tokenizer error")
    # Combine NOT + IN into NOT IN for easier parsing
    combined = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if (
            i + 1 < len(tokens)
            and t[0] == "WORD"
            and t[1] == "NOT"
            and tokens[i + 1][0] == "WORD"
            and tokens[i + 1][1] == "IN"
        ):
            combined.append(("WORD", "NOT IN", "NOT IN"))
            i += 2
        else:
            combined.append(t)
            i += 1
    return combined

class _TokenStream:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos >= len(self.tokens):
            return ("EOF", "EOF", None)
        return self.tokens[self.pos]

    def next(self):
        tok = self.peek()
        if self.pos < len(self.tokens):
            self.pos += 1
        return tok

    def match_word(self, word: str) -> bool:
        tok = self.peek()
        if tok[0] == "WORD" and tok[1] == word:
            self.next()
            return True
        return False

    def match_type(self, typ: str) -> bool:
        tok = self.peek()
        if tok[0] == typ:
            self.next()
            return True
        return False

    def expect(self, typ: str, value: str = None):
        tok = self.peek()
        if tok[0] != typ or (value is not None and tok[1] != value):
            raise ValueError(f"Expected {typ} {value or ''} but got {tok}")
        return self.next()

def _parse_identifier(ts: _TokenStream) -> str:
    tok = ts.peek()
    if tok[0] == "WORD":
        ts.next()
        return tok[2]  # original case
    raise ValueError("Identifier expected in predicate")

def _parse_value(ts: _TokenStream):
    tok = ts.peek()
    if tok[0] == "NUMBER":
        ts.next()
        text = tok[2]
        if any(c in text for c in ".eE"):
            try:
                return float(text)
            except ValueError:
                raise ValueError(f"Invalid number literal: {text}")
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"Invalid integer literal: {text}")
    elif tok[0] == "STRING":
        ts.next()
        return _unescape_string(tok[2])
    elif tok[0] == "WORD":
        if tok[1] == "TRUE":
            ts.next()
            return True
        if tok[1] == "FALSE":
            ts.next()
            return False
        if tok[1] == "NULL":
            ts.next()
            return None
        # Unquoted bare words are not allowed as values (likely a column name)
        raise ValueError(f"Unexpected bare identifier as value: {tok[2]!r}")
    else:
        raise ValueError(f"Unexpected token in value: {tok}")

def _parse_value_list(ts: _TokenStream):
    values = []
    ts.expect("LPAREN", "(")
    # Empty list allowed?
    first = True
    while True:
        if ts.match_type("RPAREN"):
            break
        if not first:
            ts.expect("COMMA", ",")
        val = _parse_value(ts)
        values.append(val)
        first = False
        if ts.match_type("RPAREN"):
            break
    return values

def _like_to_regex(pattern: str) -> re.Pattern:
    # Convert SQL LIKE pattern (% -> .*, _ -> .)
    # Escape regex metacharacters first
    regex = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "%":
            regex.append(".*")
        elif c == "_":
            regex.append(".")
        else:
            regex.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(regex) + "$")

def _parse_predicate(ts: _TokenStream) -> Callable[[Dict[str, Any]], bool]:
    field = _parse_identifier(ts)
    tok = ts.peek()

    # IN / NOT IN
    if tok[0] == "WORD" and tok[1] in ("IN", "NOT IN"):
        negate = tok[1] == "NOT IN"
        ts.next()
        values = _parse_value_list(ts)
        values_set = set(values)

        def pred(rec: Dict[str, Any]) -> bool:
            val = rec.get(field, None)
            res = val in values_set
            return (not res) if negate else res

        return pred

    # LIKE / ILIKE
    if tok[0] == "WORD" and tok[1] in ("LIKE", "ILIKE"):
        case_insensitive = tok[1] == "ILIKE"
        ts.next()
        pat_val = _parse_value(ts)
        if not isinstance(pat_val, str):
            raise ValueError("LIKE/ILIKE requires a string pattern")
        regex = _like_to_regex(pat_val)
        if case_insensitive:
            regex = re.compile(regex.pattern, re.IGNORECASE)

        def pred(rec: Dict[str, Any]) -> bool:
            val = rec.get(field, None)
            if val is None:
                return False
            return bool(regex.fullmatch(str(val)))

        return pred

    # Comparison operators
    if tok[0] == "OP":
        op_text = tok[1]
        ts.next()
        right = _parse_value(ts)
        op_map = {
            "=": operator.eq,
            "!=": operator.ne,
            "<>": operator.ne,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
        }
        if op_text not in op_map:
            raise ValueError(f"Unsupported operator: {op_text}")
        op_func = op_map[op_text]

        def pred(rec: Dict[str, Any]) -> bool:
            left = rec.get(field, None)
            try:
                return bool(op_func(left, right))
            except Exception:
                return False

        return pred

    raise ValueError("Invalid predicate near: {}".format(tok))

def _parse_unary(ts: _TokenStream) -> Callable[[Dict[str, Any]], bool]:
    if ts.match_word("NOT"):
        inner = _parse_unary(ts)

        def neg(rec: Dict[str, Any]) -> bool:
            return not inner(rec)

        return neg
    if ts.match_type("LPAREN"):
        inner = _parse_expr(ts)
        ts.expect("RPAREN", ")")
        return inner
    return _parse_predicate(ts)

def _parse_and(ts: _TokenStream) -> Callable[[Dict[str, Any]], bool]:
    left = _parse_unary(ts)

    def make_and(a, b):
        return lambda rec: a(rec) and b(rec)

    while ts.match_word("AND"):
        right = _parse_unary(ts)
        left = make_and(left, right)
    return left

def _parse_expr(ts: _TokenStream) -> Callable[[Dict[str, Any]], bool]:
    left = _parse_and(ts)

    def make_or(a, b):
        return lambda rec: a(rec) or b(rec)

    while ts.match_word("OR"):
        right = _parse_and(ts)
        left = make_or(left, right)
    return left

def _compile_where(where_str: str) -> Callable[[Dict[str, Any]], bool]:
    tokens = _tokenize_where(where_str)
    ts = _TokenStream(tokens)
    predicate = _parse_expr(ts)
    # Ensure all tokens consumed
    if ts.peek()[0] != "EOF":
        raise ValueError("Unexpected tokens after WHERE clause")
    return predicate

def _parse_select_fields(select_part: str):
    fields_part = select_part.strip()
    if fields_part == "*":
        return None  # select all
    # Split by commas
    fields = []
    for seg in fields_part.split(","):
        name = seg.strip()
        if not name:
            continue
        # Support optional AS alias; ignore alias for output key names
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_\.]*)(?:\s+AS\s+[A-Za-z_][A-Za-z0-9_\.]*)?$", name, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Invalid field in SELECT: {name!r}")
        fields.append(m.group(1))
    if not fields:
        raise ValueError("SELECT clause has no fields")
    return fields

def _parse_order_by(order_part: str):
    items = [item.strip() for item in order_part.split(",") if item.strip()]
    orderings = []
    for item in items:
        parts = item.split()
        if not parts:
            continue
        field = parts[0]
        direction = parts[1].upper() if len(parts) > 1 else "ASC"
        if direction not in ("ASC", "DESC"):
            raise ValueError(f"Invalid ORDER BY direction for {field!r}: {direction!r}")
        orderings.append((field, direction == "ASC"))
    return orderings

def _safe_sort_key(val: Any):
    if val is None:
        return (1, 0)
    # Normalize types for consistent comparisons
    if isinstance(val, bool):
        return (0, int(val))
    if isinstance(val, (int, float)):
        return (0, float(val))
    if isinstance(val, str):
        return (0, val)
    # Fallback to string representation
    return (0, str(val))

def execute_query_cmd(dataset_list: List[Dict[str, Any]], sql_query: str) -> List[Dict[str, Any]]:
    """
    Execute a simple SQL-like query over a list of dictionaries.

    Supported features:
    - SELECT fields or SELECT *
    - WHERE with operators: =, !=, <>, <, <=, >, >=, IN, NOT IN, LIKE, ILIKE
      Logical operators: NOT, AND, OR with parentheses
    - ORDER BY field [ASC|DESC], multiple keys supported

    Raises ValueError on malformed queries or execution errors.
    """
    try:
        if not isinstance(sql_query, str) or not sql_query.strip():
            raise ValueError("sql_query must be a non-empty string")

        # Parse the query into SELECT, FROM, WHERE, ORDER BY
        # FROM part is required syntactically but the table name is ignored.
        m = re.match(
            r"(?is)^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<table>[A-Za-z_][A-Za-z0-9_\.]*)(?:\s+WHERE\s+(?P<where>.+?))?(?:\s+ORDER\s+BY\s+(?P<order>.+?))?\s*$",
            sql_query,
        )
        if not m:
            raise ValueError("Malformed query: could not parse SELECT/FROM/WHERE/ORDER BY")

        select_part = m.group("select")
        where_part = m.group("where")
        order_part = m.group("order")

        # Compile WHERE predicate if present
        if where_part is not None:
            predicate = _compile_where(where_part)
            filtered = filter_data(list(dataset_list or []), predicate)
        else:
            filtered = [rec for rec in (dataset_list or []) if isinstance(rec, dict)]

        # Apply ORDER BY on filtered data (using source fields)
        if order_part:
            orderings = _parse_order_by(order_part)
            sorted_records = list(filtered)
            # Apply stable sorts from last key to first
            for field, asc in reversed(orderings):
                sorted_records.sort(
                    key=lambda rec, f=field: _safe_sort_key(rec.get(f, None)),
                    reverse=not asc,
                )
        else:
            sorted_records = list(filtered)

        # Apply SELECT
        fields = _parse_select_fields(select_part)
        if fields is None:
            # SELECT * -> return shallow copies
            result = [dict(rec) for rec in sorted_records]
        else:
            result = select_fields(sorted_records, fields)

        return result
    except Exception as e:
        raise ValueError(f"Query execution failed: {e}") from None
