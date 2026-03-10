import re
import operator
from functools import partial
from typing import Iterable, Mapping, Sequence, List, Dict, Any, Optional, Set, Tuple, Callable

_MISSING = object()


def extract_fields(
    dataset: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    conditions: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract a subset of fields from each dictionary in a dataset, with optional filtering.

    Args:
        dataset: An iterable of dictionaries (or Mapping) representing the dataset.
        fields: A sequence of field names to extract.
        conditions: An optional mapping of field -> value pairs that a record must satisfy
            to be included in the output. All conditions are matched by equality (==).

    Returns:
        A list of dictionaries, each containing only the requested fields that were present
        in the corresponding input item. Missing fields are simply omitted.

    Raises:
        TypeError: If any item in the dataset is not a Mapping, or if any field/condition
            name is not a str, or if conditions is not a Mapping.
        ValueError: If a condition references a field that does not exist in the dataset.
    """
    # Validate fields are strings
    for f in fields:
        if not isinstance(f, str):
            raise TypeError(f"Field names must be strings. Got {type(f).__name__}: {f!r}")

    # Validate conditions
    if conditions is not None:
        if not isinstance(conditions, Mapping):
            raise TypeError(f"conditions must be a mapping. Got {type(conditions).__name__}")
        for k in conditions.keys():
            if not isinstance(k, str):
                raise TypeError(f"Condition field names must be strings. Got {type(k).__name__}: {k!r}")

    # Materialize dataset and validate items are mappings
    dataset_list: List[Mapping[str, Any]] = []
    for idx, item in enumerate(dataset):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"All dataset items must be mappings (dict-like). "
                f"Item at index {idx} is {type(item).__name__}."
            )
        dataset_list.append(item)

    # If conditions provided, ensure all referenced fields exist in the dataset (in at least one record)
    if conditions:
        available_keys: Set[str] = set()
        for item in dataset_list:
            available_keys.update(item.keys())
        missing = [k for k in conditions.keys() if k not in available_keys]
        if missing:
            missing_sorted = ", ".join(sorted(missing))
            raise ValueError(f"Condition references non-existent field(s): {missing_sorted}")

    result: List[Dict[str, Any]] = []
    for item in dataset_list:
        # Apply filtering conditions (records missing a condition field are treated as non-matching)
        if conditions:
            matched = True
            for key, expected in conditions.items():
                val = item.get(key, _MISSING)
                if val is _MISSING or val != expected:
                    matched = False
                    break
            if not matched:
                continue

        # Preserve the order of fields as provided
        subset: Dict[str, Any] = {}
        for key in fields:
            if key in item:
                subset[key] = item[key]
        result.append(subset)

    return result


def handle_sql_query(records: List[Dict[str, Any]], sql_command: str) -> List[Dict[str, Any]]:
    """
    Parse and execute a simple SQL-like query over a list of dictionaries.

    Supported syntax (case-insensitive keywords):
      SELECT field1, field2 | *
      FROM records
      [WHERE <expression>]
    Expressions support:
      - Comparisons: =, !=, <, <=, >, >=
      - IN (...) and NOT IN (...)
      - LIKE 'pattern'  (SQL wildcards: % and _)
      - Parentheses for grouping
      - Logical operators: AND, OR, NOT
      - Literals: strings in '...' or "...", numbers, TRUE, FALSE, NULL
      - Comparing against another field (right-hand side IDENT) is supported

    Returns:
        List[dict]: Projected records matching the WHERE clause.

    Raises:
        ValueError: If the query is malformed or fails during execution.
    """
    try:
        if not isinstance(records, list) or not all(isinstance(r, Mapping) for r in records):
            raise ValueError("records must be a list of dictionaries (mapping objects).")
        if not isinstance(sql_command, str):
            raise ValueError("sql_command must be a string.")

        # Parse main SELECT ... FROM ... [WHERE ...]
        select_re = re.compile(
            r"^\s*SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<table>[A-Za-z_][A-Za-z0-9_]*)"
            r"(?:\s+WHERE\s+(?P<where>.+?))?\s*;?\s*$",
            re.IGNORECASE | re.DOTALL,
        )
        m = select_re.match(sql_command)
        if not m:
            raise ValueError("Malformed SQL: could not parse SELECT/FROM/WHERE structure.")

        select_part = m.group("select").strip()
        table_part = m.group("table").strip()
        where_part = m.group("where")
        if table_part.lower() != "records":
            raise ValueError("Unsupported table name. Use 'records' to refer to the given dataset.")

        # Collect available keys across all records
        available_keys: Set[str] = set()
        for rec in records:
            available_keys.update(rec.keys())

        # Parse fields
        select_all = select_part == "*"
        selected_fields: List[str] = []
        if not select_all:
            raw_fields = [f.strip() for f in select_part.split(",")]
            if not raw_fields or any(not f for f in raw_fields):
                raise ValueError("Malformed SQL: empty field in SELECT list.")
            ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
            for f in raw_fields:
                if not ident_re.match(f):
                    raise ValueError(f"Invalid field name in SELECT list: {f!r}")
                selected_fields.append(f)

            # Validate selected fields exist in dataset
            missing_select = [f for f in selected_fields if f not in available_keys]
            if missing_select:
                raise ValueError(
                    f"Selected non-existent field(s): {', '.join(sorted(missing_select))}"
                )

        # Build predicate from WHERE clause (if any)
        predicate: Callable[[Mapping[str, Any]], bool]
        referenced_fields: Set[str] = set()
        if where_part and where_part.strip():
            predicate, referenced_fields = _build_where_predicate(where_part)

            # Ensure all referenced fields exist
            missing = [f for f in referenced_fields if f not in available_keys]
            if missing:
                raise ValueError(
                    f"WHERE references non-existent field(s): {', '.join(sorted(missing))}"
                )
        else:
            predicate = lambda rec: True

        # Execute: filter
        filtered: List[Dict[str, Any]] = []
        for rec in records:
            try:
                if predicate(rec):
                    if select_all:
                        filtered.append(dict(rec))
                    else:
                        # Preserve order of selected fields; include only present keys
                        out: Dict[str, Any] = {}
                        for f in selected_fields:
                            if f in rec:
                                out[f] = rec[f]
                        filtered.append(out)
            except Exception:
                # Any unexpected error during evaluation treated as non-match
                # rather than failing the whole query
                continue

        return filtered

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to execute query: {exc}") from exc


# --- Internal utilities for WHERE parsing/evaluation ---

_Token = Tuple[str, Any]


def _tokenize(where: str) -> List[_Token]:
    token_re = re.compile(
        r"""
        (?P<SPACE>\s+)
        |(?P<LPAREN>\()
        |(?P<RPAREN>\))
        |(?P<COMMA>,)
        |(?P<OP><=|>=|!=|=|<|>)
        |(?P<STRING>'([^'\\]|\\.)*'|"([^"\\]|\\.)*")
        |(?P<NUMBER>\d+(?:\.\d+)?)
        |(?P<WORD>[A-Za-z_][A-Za-z0-9_]*)
        """,
        re.VERBOSE,
    )
    tokens: List[_Token] = []
    pos = 0
    while pos < len(where):
        m = token_re.match(where, pos)
        if not m:
            raise ValueError(f"Malformed WHERE clause near: {where[pos:pos+20]!r}")
        kind = m.lastgroup or ""
        text = m.group(kind)
        pos = m.end()
        if kind == "SPACE":
            continue
        if kind == "WORD":
            upper = text.upper()
            if upper in ("AND", "OR", "NOT", "IN", "LIKE", "TRUE", "FALSE", "NULL"):
                kind = upper
                val: Any
                if upper == "TRUE":
                    val = True
                elif upper == "FALSE":
                    val = False
                elif upper == "NULL":
                    val = None
                else:
                    val = upper
                tokens.append((kind, val))
                continue
            tokens.append(("IDENT", text))
            continue
        if kind == "NUMBER":
            if "." in text:
                tokens.append(("NUMBER", float(text)))
            else:
                tokens.append(("NUMBER", int(text)))
            continue
        if kind == "STRING":
            s = text[1:-1]
            quote = text[0]
            # Unescape \" or \' and backslash escapes
            s = s.replace(f"\\{quote}", quote).replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
            tokens.append(("STRING", s))
            continue
        if kind in ("LPAREN", "RPAREN", "COMMA", "OP"):
            tokens.append((kind, text))
            continue
        raise ValueError(f"Unsupported token in WHERE: {text!r}")
    return tokens


def _build_where_predicate(where: str) -> Tuple[Callable[[Mapping[str, Any]], bool], Set[str]]:
    tokens = _tokenize(where)
    pos = 0

    def peek() -> Optional[_Token]:
        return tokens[pos] if pos < len(tokens) else None

    def consume(expected_kind: Optional[str] = None) -> _Token:
        nonlocal pos
        tok = peek()
        if tok is None:
            raise ValueError("Unexpected end of WHERE clause.")
        if expected_kind and tok[0] != expected_kind:
            raise ValueError(f"Expected {expected_kind}, got {tok[0]} near position {pos}.")
        pos += 1
        return tok

    def parse_value() -> Tuple[Any, Optional[str]]:
        tok = peek()
        if tok is None:
            raise ValueError("Unexpected end when parsing value.")
        kind, val = tok
        if kind in ("STRING", "NUMBER", "TRUE", "FALSE", "NULL"):
            consume()
            return val, None
        if kind == "IDENT":
            consume()
            # Return marker indicating it's a field reference
            return val, "FIELD"
        if kind == "LPAREN":
            # List of values
            consume("LPAREN")
            values: List[Any] = []
            first = True
            while True:
                if peek() and peek()[0] == "RPAREN":
                    if first:
                        raise ValueError("Empty list not allowed in IN clause.")
                    break
                if not first:
                    consume("COMMA")
                v, field_marker = parse_value()
                if field_marker == "FIELD":
                    # For IN lists we only allow literals
                    raise ValueError("IN list must contain literals only.")
                values.append(v)
                first = False
                if peek() and peek()[0] == "RPAREN":
                    break
            consume("RPAREN")
            return values, None
        raise ValueError(f"Invalid value token: {tok!r}")

    def make_compare(field: str, op_func: Callable[[Any, Any], bool], rhs: Any, rhs_is_field: bool) -> Callable[[Mapping[str, Any]], bool]:
        def pred(rec: Mapping[str, Any]) -> bool:
            a = rec.get(field, _MISSING)
            if a is _MISSING:
                return False
            b = rec.get(rhs, _MISSING) if rhs_is_field else rhs
            if rhs_is_field and b is _MISSING:
                return False
            try:
                return bool(op_func(a, b))
            except Exception:
                return False
        return pred

    def make_like(field: str, pattern: str) -> Callable[[Mapping[str, Any]], bool]:
        # Convert SQL LIKE to regex
        # Escape regex special chars except for % and _
        def to_regex(p: str) -> str:
            escaped = ""
            for ch in p:
                if ch == "%":
                    escaped += ".*"
                elif ch == "_":
                    escaped += "."
                else:
                    escaped += re.escape(ch)
            return f"^{escaped}$"

        regex = re.compile(to_regex(pattern))
        # Use functools.partial to create a value tester
        tester = partial(lambda reg, s: isinstance(s, str) and reg.fullmatch(s) is not None, regex)

        def pred(rec: Mapping[str, Any]) -> bool:
            val = rec.get(field, _MISSING)
            if val is _MISSING:
                return False
            try:
                return bool(tester(val))
            except Exception:
                return False

        return pred

    def parse_comparison() -> Tuple[Callable[[Mapping[str, Any]], bool], Set[str]]:
        tok = peek()
        if not tok or tok[0] != "IDENT":
            raise ValueError("Expected field identifier at start of comparison.")
        _, field = consume("IDENT")
        referenced: Set[str] = {field}

        nxt = peek()
        if not nxt:
            raise ValueError("Incomplete comparison in WHERE clause.")

        # Handle NOT IN
        if nxt[0] == "NOT":
            consume("NOT")
            nxt = peek()
            if not nxt or nxt[0] != "IN":
                # Treat as unary NOT expression if not followed by IN
                # Rollback pos movement is complex; instead, raise a clear error
                raise ValueError("Expected IN after NOT in comparison.")
            consume("IN")
            val, marker = parse_value()
            if marker is not None or not isinstance(val, list):
                raise ValueError("NOT IN expects a parenthesized list of literals.")
            values_set = set(val)

            def pred(rec: Mapping[str, Any]) -> bool:
                v = rec.get(field, _MISSING)
                if v is _MISSING:
                    return False
                try:
                    return v not in values_set
                except Exception:
                    return False

            return pred, referenced

        # Handle IN
        if nxt[0] == "IN":
            consume("IN")
            val, marker = parse_value()
            if marker is not None or not isinstance(val, list):
                raise ValueError("IN expects a parenthesized list of literals.")
            values_set = set(val)

            # Use partial to create a membership tester
            membership = partial(lambda coll, x: x in coll, values_set)

            def pred(rec: Mapping[str, Any]) -> bool:
                v = rec.get(field, _MISSING)
                if v is _MISSING:
                    return False
                try:
                    return bool(membership(v))
                except Exception:
                    return False

            return pred, referenced

        # Handle LIKE
        if nxt[0] == "LIKE":
            consume("LIKE")
            val, marker = parse_value()
            if marker is not None or not isinstance(val, str):
                raise ValueError("LIKE expects a string literal pattern.")
            return make_like(field, val), referenced

        # Handle standard operators
        if nxt[0] == "OP":
            _, op_text = consume("OP")
            rhs, rhs_marker = parse_value()
            op_map: Dict[str, Callable[[Any, Any], bool]] = {
                "=": operator.eq,
                "!=": operator.ne,
                "<": operator.lt,
                "<=": operator.le,
                ">": operator.gt,
                ">=": operator.ge,
            }
            if op_text not in op_map:
                raise ValueError(f"Unsupported operator: {op_text!r}")
            comp_func = op_map[op_text]
            pred = make_compare(field, comp_func, rhs, rhs_is_field=(rhs_marker == "FIELD"))
            if rhs_marker == "FIELD":
                referenced.add(rhs)  # type: ignore[arg-type]
            return pred, referenced

        raise ValueError(f"Unexpected token in comparison: {nxt!r}")

    def parse_factor() -> Tuple[Callable[[Mapping[str, Any]], bool], Set[str]]:
        tok = peek()
        if tok and tok[0] == "NOT":
            consume("NOT")
            inner_pred, refs = parse_factor()
            return (lambda rec: not inner_pred(rec)), refs
        if tok and tok[0] == "LPAREN":
            consume("LPAREN")
            inner_pred, refs = parse_expr()
            consume("RPAREN")
            return inner_pred, refs
        return parse_comparison()

    def parse_term() -> Tuple[Callable[[Mapping[str, Any]], bool], Set[str]]:
        pred, refs = parse_factor()
        while True:
            tok = peek()
            if tok and tok[0] == "AND":
                consume("AND")
                right_pred, right_refs = parse_factor()
                left = pred
                pred = lambda rec, l=left, r=right_pred: l(rec) and r(rec)
                refs |= right_refs
            else:
                break
        return pred, refs

    def parse_expr() -> Tuple[Callable[[Mapping[str, Any]], bool], Set[str]]:
        pred, refs = parse_term()
        while True:
            tok = peek()
            if tok and tok[0] == "OR":
                consume("OR")
                right_pred, right_refs = parse_term()
                left = pred
                pred = lambda rec, l=left, r=right_pred: l(rec) or r(rec)
                refs |= right_refs
            else:
                break
        return pred, refs

    predicate, refs = parse_expr()
    if pos != len(tokens):
        raise ValueError("Unexpected tokens at end of WHERE clause.")
    return predicate, refs
