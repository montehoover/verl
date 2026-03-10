import re
import operator
from functools import partial
from typing import Any, Dict, List, Mapping, Sequence, Optional


def extract_fields(
    records: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    conditions: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract a subset of fields from each record in a list of dictionaries, with optional filtering.

    Args:
        records: A sequence of mapping objects (e.g., dicts) representing the dataset.
        fields: A sequence of field names to extract from each record.
        conditions: A mapping of field names to required values. Only records that meet all
            conditions are included. If any condition references a field that does not exist
            in a record, a ValueError is raised.

    Returns:
        A list of dictionaries where each dictionary contains only the specified fields
        present in the corresponding input record. Missing fields are omitted.

    Raises:
        TypeError: If any item in records is not a mapping/dict, or if conditions is not a mapping.
        ValueError: If fields is None, or if a condition references a non-existent field in a record.
    """
    if fields is None:
        raise ValueError("fields must not be None")

    if conditions is not None and not isinstance(conditions, Mapping):
        raise TypeError("conditions must be a mapping/dict if provided")

    # Deduplicate fields while preserving order
    unique_fields = list(dict.fromkeys(fields))

    cond_items = list(conditions.items()) if conditions else []

    result: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"Record at index {idx} is not a mapping/dict: {type(record).__name__}")

        # Apply filtering conditions
        include = True
        for key, expected in cond_items:
            if key not in record:
                raise ValueError(f"Condition references non-existent field '{key}' in record at index {idx}")
            if record[key] != expected:
                include = False
                break
        if not include:
            continue

        filtered = {key: record[key] for key in unique_fields if key in record}
        result.append(filtered)

    return result


def execute_query_cmd(dataset_list: Sequence[Mapping[str, Any]], sql_query: str) -> List[Dict[str, Any]]:
    """
    Execute a simple SQL-like command against a list of dictionaries.

    Supported syntax (FROM is implicit; the dataset is provided via dataset_list):
      SELECT field1, field2, ...
      SELECT *
      [WHERE <predicate> [AND <predicate> ...]]
      [ORDER BY field [ASC|DESC] [, field2 [ASC|DESC] ...]]

    Predicates support:
      - field = value
      - field != value (or <>)
      - field <, <=, >, >= value
      - field IN (value1, value2, ...)
      - field NOT IN (value1, value2, ...)

    Values can be:
      - Numbers (int or float), booleans (true/false), null/none, or quoted strings ('...' or "...").

    Returns:
        List of dictionaries representing the query result (projection happens after filtering and sorting).

    Raises:
        ValueError: If the query is malformed or cannot be processed (e.g., invalid syntax, missing fields in WHERE/ORDER BY).
    """

    if not isinstance(sql_query, str):
        raise ValueError("sql_query must be a string")

    query = sql_query.strip()
    if not query:
        raise ValueError("Empty query")

    # Allow trailing semicolon
    if query.endswith(";"):
        query = query[:-1].rstrip()

    # Parse top-level clauses: SELECT ... [WHERE ...] [ORDER BY ...]
    # WHERE is made non-greedy so ORDER BY (if present) is captured by the last group.
    m = re.match(
        r"^\s*SELECT\s+(?P<select>.*?)\s*(?:WHERE\s+(?P<where>.*?))?\s*(?:ORDER\s+BY\s+(?P<order>.*))?\s*$",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        raise ValueError("Malformed query: expected SELECT [WHERE ...] [ORDER BY ...]")

    select_part = (m.group("select") or "").strip()
    where_part = (m.group("where") or "").strip()
    order_part = (m.group("order") or "").strip()

    if not select_part:
        raise ValueError("Malformed query: missing SELECT list")

    # Parse SELECT fields
    select_all = select_part == "*"
    if not select_all:
        select_fields = [s.strip() for s in select_part.split(",") if s.strip()]
        if not select_fields:
            raise ValueError("Malformed query: empty SELECT list")
        ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        for f in select_fields:
            if not ident_re.fullmatch(f):
                raise ValueError(f"Malformed query: invalid field name in SELECT: {f!r}")
    else:
        select_fields = []

    # Helper functions for parsing WHERE predicates
    def _split_comma_list(s: str) -> List[str]:
        items: List[str] = []
        buf: List[str] = []
        quote: Optional[str] = None
        escape = False
        for ch in s:
            if escape:
                buf.append(ch)
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if quote:
                if ch == quote:
                    quote = None
                else:
                    buf.append(ch)
                continue
            if ch in ("'", '"'):
                quote = ch
                continue
            if ch == "," and not quote:
                items.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        if quote:
            raise ValueError("Malformed query: unmatched quote in list")
        tail = "".join(buf).strip()
        if tail:
            items.append(tail)
        return items

    def _unescape_quoted(s: str) -> str:
        out: List[str] = []
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == "\\" and i + 1 < len(s):
                i += 1
                out.append(s[i])
            else:
                out.append(ch)
            i += 1
        return "".join(out)

    def _parse_atom(token: str) -> Any:
        t = token.strip()
        if not t:
            raise ValueError("Malformed query: empty value")
        # Quoted string
        if (t[0] == t[-1]) and t[0] in ("'", '"') and len(t) >= 2:
            inner = t[1:-1]
            return _unescape_quoted(inner)
        # Booleans
        tl = t.lower()
        if tl == "true":
            return True
        if tl == "false":
            return False
        if tl in ("null", "none"):
            return None
        # Number (int or float)
        num_match = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", t)
        if num_match:
            if "." in t:
                try:
                    return float(t)
                except ValueError:
                    pass
            else:
                try:
                    return int(t)
                except ValueError:
                    pass
        # Fallback: treat as bare string literal
        return t

    # Build predicate functions
    predicates = []
    if where_part:
        # Split on AND (case-insensitive) not considering quoted content (simplified assumption)
        cond_tokens = re.split(r"\s+AND\s+", where_part, flags=re.IGNORECASE)
        comp_ops = {
            "=": operator.eq,
            "!=": operator.ne,
            "<>": operator.ne,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
        }
        for raw in cond_tokens:
            cond = raw.strip()
            if not cond:
                raise ValueError("Malformed query: empty predicate in WHERE")

            # IN or NOT IN
            m_in = re.match(
                r"^(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<op>NOT\s+IN|IN)\s*\((?P<vals>.*)\)\s*$",
                cond,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m_in:
                field = m_in.group("field")
                op_word = m_in.group("op").upper().replace("  ", " ")
                vals_raw = m_in.group("vals")
                items = _split_comma_list(vals_raw)
                if not items:
                    raise ValueError("Malformed query: IN list is empty")
                parsed_vals = [_parse_atom(x) for x in items]

                # Prefer set for performance when possible
                try:
                    values_container = set(parsed_vals)
                    is_set = True
                except TypeError:
                    values_container = list(parsed_vals)
                    is_set = False

                if op_word == "IN":
                    def pred_in(rec: Mapping[str, Any], f=field, cont=values_container, use_set=is_set) -> bool:
                        if f not in rec:
                            raise ValueError(f"WHERE references non-existent field '{f}'")
                        val = rec[f]
                        return (val in cont) if use_set else any(val == v for v in cont)
                    predicates.append(pred_in)
                else:
                    def pred_not_in(rec: Mapping[str, Any], f=field, cont=values_container, use_set=is_set) -> bool:
                        if f not in rec:
                            raise ValueError(f"WHERE references non-existent field '{f}'")
                        val = rec[f]
                        return (val not in cont) if use_set else all(val != v for v in cont)
                    predicates.append(pred_not_in)
                continue

            # Comparisons
            m_cmp = re.match(
                r"^(?P<field>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>=|!=|<>|<=|>=|<|>)\s*(?P<value>.+?)\s*$",
                cond,
            )
            if m_cmp:
                field = m_cmp.group("field")
                op_symbol = m_cmp.group("op")
                rhs = _parse_atom(m_cmp.group("value"))
                op_func = comp_ops[op_symbol]

                def make_cmp(f: str, cmp_func, rhs_val: Any):
                    def predicate(rec: Mapping[str, Any]) -> bool:
                        if f not in rec:
                            raise ValueError(f"WHERE references non-existent field '{f}'")
                        try:
                            return cmp_func(rec[f], rhs_val)
                        except Exception as ex:
                            raise ValueError(f"Cannot compare field '{f}' to value {rhs_val!r}: {ex}") from None
                    return predicate

                predicates.append(make_cmp(field, op_func, rhs))
                continue

            # Unsupported predicate
            raise ValueError(f"Malformed WHERE predicate: {cond!r}")

    # Apply filtering
    filtered: List[Mapping[str, Any]] = []
    for idx, rec in enumerate(dataset_list):
        if not isinstance(rec, Mapping):
            raise ValueError(f"Dataset item at index {idx} is not a mapping/dict")
        include = True
        for pred in predicates:
            if not pred(rec):
                include = False
                break
        if include:
            filtered.append(rec)

    # ORDER BY processing
    if order_part:
        order_specs: List[Dict[str, Any]] = []
        for token in [t.strip() for t in order_part.split(",") if t.strip()]:
            m_ord = re.fullmatch(
                r"(?P<field>[A-Za-z_][A-Za-z0-9_]*)(?:\s+(?P<dir>ASC|DESC))?",
                token,
                flags=re.IGNORECASE,
            )
            if not m_ord:
                raise ValueError(f"Malformed ORDER BY spec: {token!r}")
            field = m_ord.group("field")
            direction = (m_ord.group("dir") or "ASC").upper()
            order_specs.append({"field": field, "reverse": direction == "DESC"})

        # Stable sort from last to first key
        records = list(filtered)
        for spec in reversed(order_specs):
            field = spec["field"]
            reverse = spec["reverse"]

            def key_func(rec: Mapping[str, Any], f=field):
                if f not in rec:
                    raise ValueError(f"ORDER BY references non-existent field '{f}'")
                return rec[f]

            try:
                records.sort(key=key_func, reverse=reverse)
            except ValueError:
                # re-raise ValueError (from missing fields) as-is
                raise
            except TypeError as ex:
                raise ValueError(f"ORDER BY on field '{field}' failed due to non-comparable values: {ex}") from None
    else:
        records = list(filtered)

    # Projection (SELECT)
    result: List[Dict[str, Any]] = []
    if select_all:
        for rec in records:
            result.append(dict(rec))
    else:
        for rec in records:
            projected = {k: rec[k] for k in select_fields if k in rec}
            result.append(projected)

    return result
