import re
import operator
from functools import partial


class _Token:
    def __init__(self, typ, val):
        self.type = typ
        self.value = val

    def __repr__(self):
        return f"_Token({self.type!r}, {self.value!r})"


def _sql_tokenize(text):
    """
    Tokenize a simple SQL-like WHERE/ORDER expression.
    Recognizes strings (single/double quotes), numbers, identifiers, operators, commas, parentheses, and keywords.
    """
    pattern = re.compile(
        r"""
        (?P<SPACE>\s+)
      | (?P<STRING>'[^']*'|"[^"]*")
      | (?P<OP><=|>=|<>|!=|=|<|>)
      | (?P<LPAREN>\()
      | (?P<RPAREN>\))
      | (?P<COMMA>,)
      | (?P<NUMBER>\d+(?:\.\d+)?)
      | (?P<IDENT>[A-Za-z_][A-Za-z0-9_\.]*)
        """,
        re.VERBOSE | re.IGNORECASE | re.DOTALL,
    )

    pos = 0
    tokens = []
    while pos < len(text):
        m = pattern.match(text, pos)
        if not m:
            raise ValueError(f"Unable to parse near: {text[pos:pos+20]!r}")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
        if kind == "SPACE":
            continue
        if kind == "STRING":
            # Strip quotes; basic unescape not supported (keeps contents as-is)
            if val[0] == "'" and val[-1] == "'":
                val = val[1:-1]
            elif val[0] == '"' and val[-1] == '"':
                val = val[1:-1]
            tokens.append(_Token("STRING", val))
            continue
        if kind == "NUMBER":
            # Parse int if possible, else float
            if "." in val:
                try:
                    num = float(val)
                except ValueError:
                    raise ValueError(f"Invalid number literal: {val}")
            else:
                try:
                    num = int(val)
                except ValueError:
                    raise ValueError(f"Invalid number literal: {val}")
            tokens.append(_Token("NUMBER", num))
            continue
        if kind == "IDENT":
            upper = val.upper()
            if upper in ("AND", "OR", "NOT", "IN", "LIKE", "ASC", "DESC", "NULL", "TRUE", "FALSE"):
                tokens.append(_Token("KEYWORD", upper))
            else:
                tokens.append(_Token("IDENT", val))
            continue
        if kind in ("OP", "LPAREN", "RPAREN", "COMMA"):
            tokens.append(_Token(kind, val))
            continue
    # End tokens with EOF marker
    tokens.append(_Token("EOF", ""))
    return tokens


def _is_numeric_like_string(s):
    return isinstance(s, str) and re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", s) is not None


def _normalize_for_sort(value):
    """
    Normalize values to comparable keys:
    - None -> (1, 0)
    - numbers -> (0, float(value))
    - strings -> (0, lowercased string or str(value))
    - other -> (0, str(value))
    """
    if value is None:
        return (1, 0)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, str):
        # If numeric-like, treat as number for sensible ordering
        if _is_numeric_like_string(value):
            try:
                return (0, float(value))
            except Exception:
                pass
        return (0, value.lower())
    return (0, str(value).lower())


def _coerce_and_compare(opfunc, left, right):
    """
    Compare left and right using opfunc with light coercion:
    - If either is None:
        - For equality/inequality, Python semantics are fine (None == x)
        - For order ops, return False
    - If number vs string and string is numeric-like, coerce to float
    - Otherwise compare directly, raising ValueError if not comparable
    """
    order_ops = {operator.lt, operator.le, operator.gt, operator.ge}
    if left is None or right is None:
        if opfunc in order_ops:
            return False
        try:
            return opfunc(left, right)
        except Exception as e:
            raise ValueError(f"Incomparable values: {left!r} and {right!r}") from e

    # Numeric coercion
    left_is_num = isinstance(left, (int, float)) and not isinstance(left, bool)
    right_is_num = isinstance(right, (int, float)) and not isinstance(right, bool)

    if left_is_num and right_is_num:
        return opfunc(float(left), float(right))

    if left_is_num and isinstance(right, str) and _is_numeric_like_string(right):
        try:
            return opfunc(float(left), float(right))
        except Exception as e:
            raise ValueError("Failed numeric comparison") from e

    if right_is_num and isinstance(left, str) and _is_numeric_like_string(left):
        try:
            return opfunc(float(left), float(right))
        except Exception as e:
            raise ValueError("Failed numeric comparison") from e

    # Fallback direct compare
    try:
        return opfunc(left, right)
    except TypeError as e:
        raise ValueError(f"Incomparable values: {left!r} and {right!r}") from e


def _build_like_pred(field, pattern):
    # Convert SQL LIKE pattern to a Python regex: % -> .* and _ -> .
    # Escape other regex chars.
    def to_regex(pat):
        # Escape then re-introduce wildcards
        escaped = ""
        for ch in pat:
            if ch == "%":
                escaped += ".*"
            elif ch == "_":
                escaped += "."
            else:
                escaped += re.escape(ch)
        return f"^{escaped}$"

    regex = re.compile(to_regex(pattern), re.IGNORECASE | re.DOTALL)

    def pred(row):
        val = row.get(field)
        if val is None:
            return False
        return regex.match(str(val)) is not None

    return pred


def _build_in_pred(field, values, negate=False):
    # For IN we apply equality with light coercion against each candidate.
    def pred(row):
        val = row.get(field)
        result = any(_coerce_and_compare(operator.eq, val, lit) for lit in values)
        return (not result) if negate else result

    return pred


def _build_op_pred(field, op_str, literal):
    ops = {
        "=": operator.eq,
        "==": operator.eq,
        "!=": operator.ne,
        "<>": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }
    if op_str not in ops:
        raise ValueError(f"Unsupported operator: {op_str}")

    opfunc = ops[op_str]

    def pred(row):
        val = row.get(field)
        return _coerce_and_compare(opfunc, val, literal)

    return pred


def _parse_value(tokens, i):
    tok = tokens[i]
    if tok.type == "STRING":
        return tok.value, i + 1
    if tok.type == "NUMBER":
        return tok.value, i + 1
    if tok.type == "KEYWORD" and tok.value in ("NULL", "TRUE", "FALSE"):
        if tok.value == "NULL":
            return None, i + 1
        if tok.value == "TRUE":
            return True, i + 1
        if tok.value == "FALSE":
            return False, i + 1
    raise ValueError(f"Expected literal value at token {tok}")


def _parse_value_list(tokens, i):
    # Expect LPAREN value (, value)* RPAREN
    if tokens[i].type != "LPAREN":
        raise ValueError("Expected '(' after IN")
    i += 1
    values = []
    first = True
    while True:
        if tokens[i].type == "RPAREN":
            if first:
                # Empty list: IN () invalid
                raise ValueError("Empty IN list")
            i += 1
            break
        val, i = _parse_value(tokens, i)
        values.append(val)
        first = False
        if tokens[i].type == "COMMA":
            i += 1
            continue
        elif tokens[i].type == "RPAREN":
            i += 1
            break
        else:
            raise ValueError("Expected ',' or ')' in IN list")
    return values, i


def _parse_where(tokens, i0=0):
    """
    Parse WHERE into a predicate function.
    Grammar (basic):
      expr := term (OR term)*
      term := factor (AND factor)*
      factor := [NOT] simple
      simple := IDENT ( IN '(' value_list ')' | LIKE value | op value )
    Returns: (predicate_fn, next_index)
    """
    def parse_factor(i):
        neg = False
        if tokens[i].type == "KEYWORD" and tokens[i].value == "NOT":
            neg = True
            i += 1
        # IDENT
        if tokens[i].type != "IDENT":
            raise ValueError(f"Expected field name in WHERE at token {tokens[i]}")
        field = tokens[i].value
        i += 1

        # Operator or keyword
        tok = tokens[i]
        if tok.type == "KEYWORD" and tok.value == "IN":
            i += 1
            values, i = _parse_value_list(tokens, i)
            pred = _build_in_pred(field, values, negate=neg)
            return pred, i
        elif tok.type == "KEYWORD" and tok.value == "LIKE":
            i += 1
            lit, i = _parse_value(tokens, i)
            if isinstance(lit, (int, float, bool)) or lit is None:
                # Convert non-strings to string pattern
                lit = "" if lit is None else str(lit)
            pred = _build_like_pred(field, str(lit))
            if neg:
                return (lambda row, f=pred: not f(row)), i
            return pred, i
        elif tok.type == "OP":
            op_str = tok.value
            i += 1
            lit, i = _parse_value(tokens, i)
            pred = _build_op_pred(field, op_str, lit)
            if neg:
                return (lambda row, f=pred: not f(row)), i
            return pred, i
        else:
            raise ValueError(f"Expected operator in WHERE at token {tok}")

    def parse_term(i):
        pred, i = parse_factor(i)
        while tokens[i].type == "KEYWORD" and tokens[i].value == "AND":
            i += 1
            right_pred, i = parse_factor(i)
            left = pred
            pred = lambda row, l=left, r=right_pred: (l(row) and r(row))
        return pred, i

    def parse_expr(i):
        pred, i = parse_term(i)
        while tokens[i].type == "KEYWORD" and tokens[i].value == "OR":
            i += 1
            right_pred, i = parse_term(i)
            left = pred
            pred = lambda row, l=left, r=right_pred: (l(row) or r(row))
        return pred, i

    pred, i_end = parse_expr(i0)
    return pred, i_end


def _parse_query(sql_query):
    """
    Parse query into components: select_fields (list or None for *), where_pred (callable or None),
    order_by (list of (field, asc_bool))
    """
    if not isinstance(sql_query, str) or not sql_query.strip():
        raise ValueError("Query must be a non-empty string")

    # Extract SELECT, FROM, optional WHERE, optional ORDER BY
    pattern = re.compile(
        r"""
        ^\s*SELECT\s+(?P<select>.+?)\s+
        FROM\s+(?P<from>\w+)
        (?:\s+WHERE\s+(?P<where>.+?))?
        (?:\s+ORDER\s+BY\s+(?P<order>.+))?
        \s*$
        """,
        re.IGNORECASE | re.VERBOSE | re.DOTALL,
    )
    m = pattern.match(sql_query)
    if not m:
        raise ValueError("Invalid query format. Expected: SELECT ... FROM <name> [WHERE ...] [ORDER BY ...]")

    select_part = m.group("select").strip()
    where_part = m.group("where")
    order_part = m.group("order")

    # Parse SELECT fields
    if select_part == "*":
        select_fields = None
    else:
        fields = [f.strip() for f in select_part.split(",")]
        if not all(f for f in fields):
            raise ValueError("Invalid SELECT field list")
        select_fields = fields

    # Parse WHERE
    where_pred = None
    if where_part:
        tokens = _sql_tokenize(where_part.strip())
        where_pred, idx = _parse_where(tokens, 0)
        # We allow trailing EOF only
        if tokens[idx].type != "EOF":
            # Skip trailing spaces/comments are handled; anything else is error
            # But allow redundant EOF
            if not (tokens[idx].type == "EOF"):
                raise ValueError("Unexpected tokens at end of WHERE clause")

    # Parse ORDER BY
    order_by = []
    if order_part:
        # Split by commas not within quotes/parentheses by tokenizing
        tokens = _sql_tokenize(order_part.strip())
        i = 0
        while True:
            if tokens[i].type == "EOF":
                break
            if tokens[i].type != "IDENT":
                raise ValueError(f"Expected field name in ORDER BY at token {tokens[i]}")
            field = tokens[i].value
            i += 1
            asc = True
            if tokens[i].type == "KEYWORD" and tokens[i].value in ("ASC", "DESC"):
                asc = tokens[i].value == "ASC"
                i += 1
            order_by.append((field, asc))
            if tokens[i].type == "COMMA":
                i += 1
                continue
            elif tokens[i].type == "EOF":
                break
            else:
                raise ValueError(f"Unexpected token in ORDER BY: {tokens[i]}")

    return select_fields, where_pred, order_by


def execute_query_cmd(dataset_list, sql_query):
    """
    Execute a basic SQL-like query over dataset_list (list of dicts).

    Args:
        dataset_list: list[dict], each dict is a record.
        sql_query: string, e.g. "SELECT name, age FROM data WHERE age > 25 ORDER BY age DESC"

    Returns:
        list[dict]: result set

    Raises:
        ValueError: on invalid query or processing errors.
    """
    if not isinstance(dataset_list, list) or not all(isinstance(x, dict) for x in dataset_list):
        raise ValueError("dataset_list must be a list of dictionaries")

    select_fields, where_pred, order_by = _parse_query(sql_query)

    # Work on a copy to avoid mutating caller's list during sorting
    rows = list(dataset_list)

    # WHERE filtering
    if where_pred is not None:
        try:
            rows = [r for r in rows if where_pred(r)]
        except ValueError:
            # Re-raise ValueError from predicate unchanged
            raise
        except Exception as e:
            raise ValueError(f"Failed to apply WHERE clause: {e}") from e

    # ORDER BY: stable multi-key sort by iterating from last to first
    if order_by:
        try:
            for field, asc in reversed(order_by):
                rows.sort(
                    key=lambda rec, f=field: _normalize_for_sort(rec.get(f)),
                    reverse=not asc,
                )
        except Exception as e:
            raise ValueError(f"Failed to apply ORDER BY: {e}") from e

    # SELECT projection
    if select_fields is None:
        # SELECT *
        result = [dict(r) for r in rows]
    else:
        # Return requested fields; if missing in a row, value is None
        result = [{f: r.get(f, None) for f in select_fields} for r in rows]

    return result
