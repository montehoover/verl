import re
import operator
import logging
from functools import partial
from pprint import pformat

class _Token:
    def __init__(self, typ, value, pos):
        self.type = typ
        self.value = value
        self.pos = pos

    def __repr__(self):
        return f"_Token({self.type}, {self.value}, {self.pos})"


def _tokenize_where(expr):
    token_spec = [
        ("SKIP", r"[ \t\r\n]+"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("COMMA", r","),
        ("OP", r"<=|>=|<>|!=|=|<|>"),
        ("STRING", r"'(?:\\.|[^'])*'"),
        ("NUMBER", r"\d+(?:\.\d+)?"),
        ("NAME", r"[A-Za-z_][A-Za-z0-9_\.]*"),
    ]
    tok_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in token_spec)
    keywords = {"AND", "OR", "NOT", "IS", "NULL", "IN", "LIKE", "TRUE", "FALSE"}
    tokens = []
    for m in re.finditer(tok_regex, expr):
        kind = m.lastgroup
        value = m.group()
        pos = m.start()
        if kind == "SKIP":
            continue
        if kind == "NAME":
            upper_val = value.upper()
            if upper_val in keywords:
                kind = upper_val
                value = upper_val
        tokens.append(_Token(kind, value, pos))
    return tokens


def _unescape_sql_string(s):
    # Strip surrounding single quotes and unescape common sequences
    assert s[0] == "'" and s[-1] == "'"
    inner = s[1:-1]
    # Replace escaped sequences
    inner = inner.replace(r"\'", "'").replace(r"\\", "\\")
    inner = inner.replace(r"\n", "\n").replace(r"\t", "\t").replace(r"\r", "\r")
    return inner


def _to_number(value_str):
    if "." in value_str:
        try:
            return float(value_str)
        except ValueError:
            return value_str
    try:
        return int(value_str)
    except ValueError:
        return value_str


def _get_field_value(record, field):
    # Support dotted paths for nested dicts
    parts = field.split(".")
    cur = record
    for p in parts:
        if not isinstance(cur, dict):
            return None
        if p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def _coerce_for_compare(lhs, rhs):
    # Try to coerce rhs to type of lhs for meaningful comparison
    if lhs is None:
        return None, None
    if isinstance(lhs, bool):
        # Consider 'true'/'false' strings or 1/0 numbers
        if isinstance(rhs, str):
            low = rhs.strip().lower()
            if low in ("true", "1"):
                return lhs, True
            if low in ("false", "0"):
                return lhs, False
        if isinstance(rhs, (int, float)):
            return lhs, bool(rhs)
    if isinstance(lhs, (int, float)):
        if isinstance(rhs, (int, float)):
            return lhs, rhs
        if isinstance(rhs, str):
            try:
                if "." in rhs:
                    return lhs, float(rhs)
                return lhs, int(rhs)
            except ValueError:
                return lhs, rhs
    if isinstance(lhs, str):
        if rhs is None:
            return lhs, None
        return lhs, str(rhs)
    return lhs, rhs


def _like_to_regex(pattern):
    # Convert SQL LIKE pattern to Python regex pattern
    # % -> .*, _ -> .
    # Escape regex metacharacters except SQL wildcards
    regex = ""
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "%":
            regex += ".*"
        elif c == "_":
            regex += "."
        else:
            regex += re.escape(c)
        i += 1
    return f"^{regex}$"


class _WhereParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self, expected_type=None):
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of WHERE clause")
        if expected_type and tok.type != expected_type:
            raise ValueError(f"Expected {expected_type} at position {tok.pos}, got {tok.type}")
        self.pos += 1
        return tok

    def parse(self):
        if not self.tokens:
            # No tokens: empty predicate: always True
            return lambda rec: True
        expr = self._parse_expression()
        return expr

    def _parse_expression(self):
        left = self._parse_term()
        while True:
            tok = self._peek()
            if tok and tok.type == "OR":
                self._consume("OR")
                right = self._parse_term()
                left = self._combine_or(left, right)
            else:
                break
        return left

    def _parse_term(self):
        left = self._parse_factor()
        while True:
            tok = self._peek()
            if tok and tok.type == "AND":
                self._consume("AND")
                right = self._parse_factor()
                left = self._combine_and(left, right)
            else:
                break
        return left

    def _parse_factor(self):
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of WHERE clause")
        if tok.type == "NOT":
            self._consume("NOT")
            inner = self._parse_factor()
            return lambda rec, inner=inner: not inner(rec)
        if tok.type == "LPAREN":
            self._consume("LPAREN")
            inner = self._parse_expression()
            self._consume("RPAREN")
            return inner
        return self._parse_comparison()

    def _parse_comparison(self):
        # identifier first
        left_tok = self._consume()
        if left_tok.type not in ("NAME",):
            raise ValueError(f"Expected identifier at position {left_tok.pos}")
        field = left_tok.value

        tok = self._peek()
        if tok is None:
            raise ValueError("Incomplete comparison in WHERE clause")

        # IS [NOT] NULL
        if tok.type == "IS":
            self._consume("IS")
            not_tok = self._peek()
            is_not = False
            if not_tok and not_tok.type == "NOT":
                self._consume("NOT")
                is_not = True
            null_tok = self._consume()
            if null_tok.type != "NULL":
                raise ValueError(f"Expected NULL at position {null_tok.pos}")
            if is_not:
                return lambda rec, f=field: _get_field_value(rec, f) is not None
            return lambda rec, f=field: _get_field_value(rec, f) is None

        # IN or NOT IN
        if tok.type == "NOT":
            # possibly NOT IN
            self._consume("NOT")
            in_tok = self._consume()
            if in_tok.type != "IN":
                raise ValueError(f"Unexpected token {in_tok.type} after NOT at position {in_tok.pos}")
            negate = True
        else:
            negate = False

        tok = self._peek()
        if tok and tok.type == "IN":
            self._consume("IN")
            self._consume("LPAREN")
            values = []
            while True:
                val_tok = self._consume()
                if val_tok.type == "STRING":
                    values.append(_unescape_sql_string(val_tok.value))
                elif val_tok.type == "NUMBER":
                    values.append(_to_number(val_tok.value))
                elif val_tok.type == "NULL":
                    values.append(None)
                elif val_tok.type in ("TRUE", "FALSE"):
                    values.append(True if val_tok.type == "TRUE" else False)
                else:
                    raise ValueError(f"Expected literal in IN() at position {val_tok.pos}")
                sep = self._peek()
                if sep and sep.type == "COMMA":
                    self._consume("COMMA")
                    continue
                elif sep and sep.type == "RPAREN":
                    self._consume("RPAREN")
                    break
                elif sep is None:
                    raise ValueError("Unterminated IN() list")
                else:
                    raise ValueError(f"Unexpected token {sep.type} in IN() at position {sep.pos}")

            if negate:
                return lambda rec, f=field, vals=tuple(values): _get_field_value(rec, f) not in vals
            return lambda rec, f=field, vals=tuple(values): _get_field_value(rec, f) in vals

        # LIKE
        if tok.type == "LIKE":
            self._consume("LIKE")
            pat_tok = self._consume()
            if pat_tok.type != "STRING":
                raise ValueError(f"Expected string literal for LIKE at position {pat_tok.pos}")
            pattern = _unescape_sql_string(pat_tok.value)
            regex = re.compile(_like_to_regex(pattern), re.IGNORECASE)
            return lambda rec, f=field, rx=regex: isinstance(_get_field_value(rec, f), str) and (rx.match(_get_field_value(rec, f) or "") is not None)

        # Standard comparison
        op_tok = self._consume()
        if op_tok.type != "OP":
            raise ValueError(f"Expected operator at position {op_tok.pos}")
        right_tok = self._consume()
        if right_tok.type == "STRING":
            right_val = _unescape_sql_string(right_tok.value)
        elif right_tok.type == "NUMBER":
            right_val = _to_number(right_tok.value)
        elif right_tok.type == "NULL":
            right_val = None
        elif right_tok.type in ("TRUE", "FALSE"):
            right_val = True if right_tok.type == "TRUE" else False
        else:
            raise ValueError(f"Expected literal at position {right_tok.pos}")

        op_map = {
            "=": operator.eq,
            "!=": operator.ne,
            "<>": operator.ne,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
        }
        if op_tok.value not in op_map:
            raise ValueError(f"Unsupported operator {op_tok.value} at position {op_tok.pos}")
        op_func = op_map[op_tok.value]

        def comparator(rec, f=field, rv=right_val, cmp=op_func):
            lv = _get_field_value(rec, f)
            if lv is None or rv is None:
                # Only equality/inequality meaningful with NULL
                if cmp == operator.eq:
                    return lv is None and rv is None
                if cmp == operator.ne:
                    return (lv is None) != (rv is None)
                return False
            lvc, rvc = _coerce_for_compare(lv, rv)
            try:
                return cmp(lvc, rvc)
            except Exception:
                # Fallback: compare as strings
                try:
                    return cmp(str(lv), str(rv))
                except Exception:
                    return False

        return comparator

    @staticmethod
    def _combine_and(a, b):
        return lambda rec, a=a, b=b: a(rec) and b(rec)

    @staticmethod
    def _combine_or(a, b):
        return lambda rec, a=a, b=b: a(rec) or b(rec)


def _split_csv(s):
    # Split by commas not inside single quotes or parentheses
    items = []
    buf = []
    in_str = False
    paren = 0
    i = 0
    while i < len(s):
        c = s[i]
        if in_str:
            buf.append(c)
            if c == "\\" and i + 1 < len(s):
                buf.append(s[i + 1])
                i += 2
                continue
            if c == "'":
                in_str = False
            i += 1
            continue
        if c == "'":
            in_str = True
            buf.append(c)
            i += 1
            continue
        if c == "(":
            paren += 1
            buf.append(c)
            i += 1
            continue
        if c == ")":
            paren = max(0, paren - 1)
            buf.append(c)
            i += 1
            continue
        if c == "," and not in_str and paren == 0:
            items.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    if buf:
        items.append("".join(buf).strip())
    return [it for it in items if it]


def _parse_select_clause(select_part):
    select_part = select_part.strip()
    if select_part == "*":
        return ["*"]
    cols = _split_csv(select_part)
    # Normalize identifiers
    out = []
    for c in cols:
        # Allow simple aliases "col AS alias" or "col alias" (optional)
        m = re.match(r"(?i)^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(?:AS\s+([A-Za-z_][A-Za-z0-9_\.]*))?\s*$", c)
        if not m:
            # If not matching alias pattern, try bare identifier
            m2 = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*$", c)
            if not m2:
                raise ValueError(f"Invalid SELECT column: {c}")
            src = m2.group(1)
            alias = None
        else:
            src = m.group(1)
            alias = m.group(2)
        out.append((src, alias or src))
    return out


def _parse_order_by(order_part):
    order_part = order_part.strip()
    if not order_part:
        return []
    items = _split_csv(order_part)
    result = []
    for item in items:
        m = re.match(r"(?i)^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*(ASC|DESC)?\s*$", item)
        if not m:
            raise ValueError(f"Invalid ORDER BY item: {item}")
        col = m.group(1)
        direction = (m.group(2) or "ASC").upper()
        result.append((col, direction))
    return result


def _parse_sql(sql_statement):
    if not isinstance(sql_statement, str) or not sql_statement.strip():
        raise ValueError("sql_statement must be a non-empty string")
    sql = sql_statement.strip().rstrip(";").strip()
    # Grammar: SELECT <cols> [WHERE <expr>] [ORDER BY <items>]
    # Use non-greedy to capture where and order by
    regex = re.compile(
        r"(?is)^\s*SELECT\s+(?P<select>.+?)\s*(?:WHERE\s+(?P<where>.+?))?\s*(?:ORDER\s+BY\s+(?P<order>.+))?\s*$"
    )
    m = regex.match(sql)
    if not m:
        raise ValueError("Query must start with SELECT and be correctly formed")
    select_part = m.group("select") or ""
    where_part = m.group("where")
    order_part = m.group("order")

    select_cols = _parse_select_clause(select_part)
    predicate = None
    if where_part:
        tokens = _tokenize_where(where_part)
        parser = _WhereParser(tokens)
        predicate = parser.parse()
    else:
        predicate = lambda rec: True
    order_by = _parse_order_by(order_part) if order_part else []
    where_text = where_part.strip() if where_part else ""
    return select_cols, predicate, order_by, where_text


def _project_record(record, select_cols):
    if select_cols == ["*"]:
        # Return a shallow copy to avoid accidental mutation
        return dict(record)
    projected = {}
    for src, alias in select_cols:
        projected[alias] = _get_field_value(record, src)
    return projected


def _sort_records(records, order_by):
    if not order_by:
        return records
    sorted_records = list(records)
    for col, direction in reversed(order_by):
        reverse = direction == "DESC"
        # Use stable sort; Python will raise if incomparable types; handle by converting to tuple key
        def key(rec, c=col):
            v = _get_field_value(rec, c)
            is_none = v is None
            return (is_none, v)
        try:
            sorted_records.sort(key=key, reverse=reverse)
        except TypeError:
            # Fallback: stringify non-None values
            def key_str(rec, c=col):
                v = _get_field_value(rec, c)
                is_none = v is None
                return (is_none, "" if v is None else str(v))
            sorted_records.sort(key=key_str, reverse=reverse)
    return sorted_records


# -------------------------
# Pipeline-friendly helpers
# -------------------------

def parse_sql_statement(sql_statement):
    """
    Pure function: parse SQL-like statement into a query plan dict.
    Returns a dict with keys: 'select', 'predicate', 'order_by', 'where_text', 'raw_sql'
    """
    select_cols, predicate, order_by, where_text = _parse_sql(sql_statement)
    raw_sql = sql_statement.strip().rstrip(";").strip() if isinstance(sql_statement, str) else ""
    return {
        "select": select_cols,
        "predicate": predicate,
        "order_by": order_by,
        "where_text": where_text,
        "raw_sql": raw_sql,
    }


def apply_where(records, predicate):
    """
    Pure function: filters records using provided predicate.
    """
    return [rec for rec in records if predicate(rec)]


def apply_sort(records, order_by):
    """
    Pure function: sorts records according to order_by specification.
    """
    return _sort_records(records, order_by)


def apply_projection(records, select_cols):
    """
    Pure function: projects records based on select columns.
    """
    return [_project_record(rec, select_cols) for rec in records]


def execute_query(dataset_records, query_plan):
    """
    Pure function: execute a query plan against dataset_records using a pipeline of steps.
    """
    steps = [
        partial(apply_where, predicate=query_plan["predicate"]),
        partial(apply_sort, order_by=query_plan["order_by"]),
        partial(apply_projection, select_cols=query_plan["select"]),
    ]
    result = dataset_records
    for step in steps:
        result = step(result)
    return result


# -------------------------
# Logging helpers
# -------------------------

def _get_logger():
    logger = logging.getLogger("sql_like_processor")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _describe_select(select_cols):
    if select_cols == ["*"]:
        return "* (all columns)"
    parts = []
    for src, alias in select_cols:
        if alias == src:
            parts.append(src)
        else:
            parts.append(f"{src} AS {alias}")
    return ", ".join(parts)


def _describe_order(order_by):
    if not order_by:
        return "(none)"
    return ", ".join(f"{col} {direction}" for col, direction in order_by)


def _format_result(records, max_items=50):
    if isinstance(records, list) and len(records) > max_items:
        preview = pformat(records[:max_items])
        return f"{preview}\n... and {len(records) - max_items} more"
    return pformat(records)


def process_sql_request(dataset_records, sql_statement):
    """
    Execute a simple SQL-like query against a list of dictionaries.

    Supported grammar (case-insensitive keywords):
      SELECT <col1, col2, ... | *>
      [WHERE <expr>]
      [ORDER BY col1 [ASC|DESC], col2 [ASC|DESC], ...]

    WHERE expressions support:
      - Comparison: =, !=, <>, <, <=, >, >= with string/number/boolean/NULL literals
      - LIKE with % (any chars) and _ (single char), case-insensitive
      - IN (...) and NOT IN (...)
      - IS NULL and IS NOT NULL
      - AND, OR, NOT, parentheses

    Returns:
      List[dict]: projected records

    Raises:
      ValueError: for malformed queries or execution errors
    """
    logger = _get_logger()
    try:
        if not isinstance(dataset_records, list):
            raise ValueError("dataset_records must be a list of dictionaries")
        for idx, rec in enumerate(dataset_records):
            if not isinstance(rec, dict):
                raise ValueError(f"dataset_records[{idx}] is not a dictionary")

        logger.info("Received query:\n%s", (sql_statement.strip().rstrip(";").strip() if isinstance(sql_statement, str) else str(sql_statement)))
        query_plan = parse_sql_statement(sql_statement)
        logger.info(
            "Plan:\n- SELECT: %s\n- WHERE: %s\n- ORDER BY: %s",
            _describe_select(query_plan["select"]),
            (query_plan.get("where_text") or "(none)"),
            _describe_order(query_plan["order_by"]),
        )

        # Execute pipeline with step-by-step logging
        logger.info("Starting pipeline with %d input records", len(dataset_records))

        filtered = apply_where(dataset_records, query_plan["predicate"])
        logger.info("WHERE step: %d -> %d records", len(dataset_records), len(filtered))

        sorted_recs = apply_sort(filtered, query_plan["order_by"])
        logger.info("ORDER BY step: %s", _describe_order(query_plan["order_by"]))

        result = apply_projection(sorted_recs, query_plan["select"])
        logger.info("SELECT step: projecting %s", _describe_select(query_plan["select"]))

        logger.info("Final result: %d records\n%s", len(result), _format_result(result))
        return result
    except ValueError as ve:
        logger.error("ValueError while processing query: %s", ve)
        raise
    except Exception as e:
        logger.exception("Failed to process query due to unexpected error")
        raise ValueError(f"Failed to process query: {e}")
