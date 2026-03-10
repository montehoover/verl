import re
import operator
from functools import partial
from typing import Iterable, Mapping, Any, Sequence, List, Dict, Optional


def extract_fields(
    records: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    conditions: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract specified fields from each record in an iterable of dictionaries,
    with optional filtering based on exact-match conditions.

    - Only includes keys that are both in the provided 'fields' list and present in each record.
    - The order of keys in each output dict follows the order provided in 'fields'.
    - Input records are not modified.
    - If 'conditions' is provided, only records that match all key==value pairs are included.
    - Raises ValueError if any condition references a field that does not exist in the dataset.

    Args:
        records: An iterable of dictionaries (mappings) representing the dataset.
        fields: A sequence of field names (strings) to extract from each record.
        conditions: Optional mapping of field names to values; records must match all to be included.

    Returns:
        A list of dictionaries with only the specified fields for each record that matches conditions.

    Raises:
        TypeError: If a record is not a mapping, fields is not a sequence of strings,
                   or conditions is not a mapping of strings to values.
        ValueError: If a condition references a non-existent field in the dataset.
    """
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        raise TypeError("fields must be a sequence of field names (e.g., list or tuple)")
    for f in fields:
        if not isinstance(f, str):
            raise TypeError("each field name in 'fields' must be a string")

    if conditions is None:
        conditions_map: Mapping[str, Any] = {}
    else:
        if not isinstance(conditions, Mapping):
            raise TypeError("conditions must be a mapping (e.g., dict) of field names to values")
        for k in conditions.keys():
            if not isinstance(k, str):
                raise TypeError("each key in 'conditions' must be a string (field name)")
        conditions_map = conditions

    result: List[Dict[str, Any]] = []
    seen_fields: set[str] = set()
    _MISSING = object()

    for idx, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise TypeError(f"record at index {idx} is not a mapping/dict")
        seen_fields.update(rec.keys())

        # Check if record matches all conditions (exact equality); missing keys fail the match
        matches = True
        for ck, cv in conditions_map.items():
            if rec.get(ck, _MISSING) != cv:
                matches = False
                break

        if matches:
            subset = {key: rec[key] for key in fields if key in rec}
            result.append(subset)

    # Validate that all condition keys exist in the dataset
    missing_condition_fields = [k for k in conditions_map.keys() if k not in seen_fields]
    if missing_condition_fields:
        missing_str = ", ".join(sorted(missing_condition_fields))
        raise ValueError(f"Condition references non-existent field(s): {missing_str}")

    return result


def run_sql_query(records: Iterable[Mapping[str, Any]], command: str) -> List[Dict[str, Any]]:
    """
    Execute a very small SQL-like query against a list of dictionaries.

    Supported features (case-insensitive keywords):
      - SELECT field1, field2, ... | SELECT *
      - Optional FROM <identifier> (ignored, for syntactic familiarity)
      - Optional WHERE with AND-only conditions:
            field = value
            field != value
            field < value
            field <= value
            field > value
            field >= value
        Values can be unquoted literals (numbers, true/false, null/none) or quoted strings ('...' or "...").
      - Optional ORDER BY field [ASC|DESC], field2 [ASC|DESC], ...

    Returns:
        List[Dict[str, Any]]: query results as a list of dictionaries.

    Raises:
        ValueError: if the query is malformed, references unknown fields, or cannot be executed.
        TypeError: if any input record is not a mapping/dict.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty SQL-like string")

    # Materialize records and validate type
    materialized: List[Mapping[str, Any]] = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise TypeError(f"record at index {idx} is not a mapping/dict")
        materialized.append(rec)

    # Collect dataset-wide field names
    all_fields_order: List[str] = []
    all_fields_set: set = set()
    for rec in materialized:
        for k in rec.keys():
            if k not in all_fields_set:
                all_fields_set.add(k)
                all_fields_order.append(k)

    # Regex to parse SELECT / WHERE / ORDER BY, with optional FROM; case-insensitive, forgiving whitespace, optional trailing semicolon
    sql_re = re.compile(
        r"""
        ^\s*SELECT\s+
            (?P<select>.*?)(?=\s+(FROM|WHERE|ORDER\s+BY)\b|;|\s*$)
        (?:\s+FROM\s+(?P<from>[A-Za-z_][A-Za-z0-9_]*)\b)?
        (?:\s+WHERE\s+(?P<where>.*?)(?=\s+ORDER\s+BY\b|;|\s*$))?
        (?:\s+ORDER\s+BY\s+(?P<order>.*?)(?=;|\s*$))?
        \s*;?\s*$
        """,
        re.IGNORECASE | re.VERBOSE | re.DOTALL,
    )

    m = sql_re.match(command)
    if not m:
        raise ValueError("Malformed SQL-like command")

    select_str = (m.group("select") or "").strip()
    where_str = (m.group("where") or "").strip()
    order_str = (m.group("order") or "").strip()

    # Helper functions
    def _split_by_commas_outside_quotes(s: str) -> List[str]:
        parts: List[str] = []
        buf: List[str] = []
        in_quote: Optional[str] = None
        i = 0
        while i < len(s):
            ch = s[i]
            if in_quote:
                buf.append(ch)
                if ch == in_quote:
                    in_quote = None
                elif ch == "\\" and i + 1 < len(s):
                    # Skip escaped char inside quotes
                    i += 1
                    buf.append(s[i])
            else:
                if ch in ("'", '"'):
                    in_quote = ch
                    buf.append(ch)
                elif ch == ",":
                    parts.append("".join(buf).strip())
                    buf = []
                else:
                    buf.append(ch)
            i += 1
        if buf or s.endswith(","):
            parts.append("".join(buf).strip())
        return [p for p in parts if p != ""]

    def _split_by_and_outside_quotes(s: str) -> List[str]:
        # Split by 'AND' (case-insensitive) when surrounded by whitespace or boundaries.
        parts: List[str] = []
        buf: List[str] = []
        in_quote: Optional[str] = None
        i = 0
        kw = "AND"
        L = len(kw)
        while i < len(s):
            if in_quote:
                ch = s[i]
                buf.append(ch)
                if ch == in_quote:
                    in_quote = None
                elif ch == "\\" and i + 1 < len(s):
                    i += 1
                    buf.append(s[i])
                i += 1
                continue

            # Not in quote: check for AND with boundaries
            if i + L <= len(s) and s[i:i+L].upper() == kw:
                prev_ok = i == 0 or s[i-1].isspace()
                next_ok = i + L == len(s) or s[i+L].isspace()
                if prev_ok and next_ok:
                    parts.append("".join(buf).strip())
                    buf = []
                    i += L
                    continue

            ch = s[i]
            if ch in ("'", '"'):
                in_quote = ch
            buf.append(ch)
            i += 1
        if buf:
            parts.append("".join(buf).strip())
        return [p for p in parts if p != ""]

    def _parse_literal(token: str) -> Any:
        t = token.strip()
        if not t:
            raise ValueError("Empty literal in WHERE clause")
        if (t[0] == t[-1]) and t[0] in ("'", '"') and len(t) >= 2:
            # Strip quotes; minimal unescape of same-quote escape (\' or \")
            quote = t[0]
            inner = t[1:-1]
            inner = inner.replace("\\" + quote, quote).replace("\\\\", "\\")
            return inner
        low = t.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low in ("null", "none"):
            return None
        # Try int, then float
        try:
            return int(t)
        except ValueError:
            try:
                return float(t)
            except ValueError:
                # Bare word -> string
                return t

    OPS = {
        "=": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }

    def _find_operator(cond: str) -> Optional[tuple[int, str]]:
        # return (index, operator_string) or None
        i = 0
        in_quote: Optional[str] = None
        while i < len(cond):
            ch = cond[i]
            if in_quote:
                if ch == in_quote:
                    in_quote = None
                elif ch == "\\":
                    i += 1  # skip escaped
            else:
                # Check two-char ops first
                if i + 2 <= len(cond):
                    two = cond[i:i+2]
                    if two in ("<=", ">=", "!="):
                        return i, two
                # Then single-char ops
                if ch in ("=", "<", ">"):
                    return i, ch
                if ch in ("'", '"'):
                    in_quote = ch
            i += 1
        return None

    # Parse SELECT list
    select_fields: List[str]
    if select_str == "*":
        select_fields = ["*"]
    else:
        select_fields = [f.strip() for f in _split_by_commas_outside_quotes(select_str)]
        if not select_fields:
            raise ValueError("SELECT clause must specify at least one field or *")
        # Validate identifiers are simple field names
        ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        for f in select_fields:
            if not ident_re.match(f):
                raise ValueError(f"Invalid field identifier in SELECT: {f!r}")

    # Parse WHERE conditions
    conditions: List[tuple[str, Any, Any]] = []  # (field, op_func, value)
    if where_str:
        tokens = _split_by_and_outside_quotes(where_str)
        if not tokens:
            raise ValueError("Malformed WHERE clause")
        for tok in tokens:
            found = _find_operator(tok)
            if not found:
                raise ValueError(f"Malformed condition in WHERE: {tok!r}")
            idx, op_sym = found
            left = tok[:idx].strip()
            right = tok[idx + len(op_sym):].strip()
            if not left or not right:
                raise ValueError(f"Malformed condition in WHERE: {tok!r}")
            # Validate field identifier
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", left):
                raise ValueError(f"Invalid field identifier in WHERE: {left!r}")
            if op_sym not in OPS:
                raise ValueError(f"Unsupported operator in WHERE: {op_sym!r}")
            value = _parse_literal(right)
            conditions.append((left, OPS[op_sym], value))

    # Verify WHERE fields exist in dataset (at least somewhere)
    cond_fields = [fld for (fld, _op, _val) in conditions]
    missing_where = sorted(set(f for f in cond_fields if f not in all_fields_set))
    if missing_where:
        raise ValueError(f"WHERE references unknown field(s): {', '.join(missing_where)}")

    # Parse ORDER BY
    order_by: List[tuple[str, bool]] = []  # (field, desc)
    if order_str:
        items = _split_by_commas_outside_quotes(order_str)
        if not items:
            raise ValueError("Malformed ORDER BY clause")
        for item in items:
            m_item = re.match(
                r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(ASC|DESC)?\s*$",
                item,
                flags=re.IGNORECASE,
            )
            if not m_item:
                raise ValueError(f"Malformed ORDER BY item: {item!r}")
            fld = m_item.group(1)
            dir_tok = (m_item.group(2) or "ASC").upper()
            desc = dir_tok == "DESC"
            order_by.append((fld, desc))

        # Validate order fields exist in dataset
        missing_order = sorted(set(f for f, _ in order_by if f not in all_fields_set))
        if missing_order:
            raise ValueError(f"ORDER BY references unknown field(s): {', '.join(missing_order)}")

    # Validate SELECT fields exist in dataset (unless '*')
    if select_fields != ["*"]:
        missing_select = sorted(set(f for f in select_fields if f not in all_fields_set))
        if missing_select:
            raise ValueError(f"SELECT references unknown field(s): {', '.join(missing_select)}")

    # Apply WHERE filtering
    _MISSING = object()

    def record_matches(rec: Mapping[str, Any]) -> bool:
        for fld, op_func, val in conditions:
            rec_val = rec.get(fld, _MISSING)
            if rec_val is _MISSING:
                return False
            try:
                if not op_func(rec_val, val):
                    return False
            except Exception:
                # Type mismatch or non-comparable values -> treat as non-match
                return False
        return True

    filtered: List[Mapping[str, Any]] = [r for r in materialized if record_matches(r)]

    # Apply ORDER BY (stable, multi-key)
    if order_by:
        # Ensure all filtered records have required fields
        for fld, _ in order_by:
            for idx, r in enumerate(filtered):
                if fld not in r:
                    raise ValueError(f"ORDER BY field {fld!r} missing in record at index {idx}")
        try:
            for fld, desc in reversed(order_by):
                filtered.sort(key=lambda rec: rec[fld], reverse=desc)
        except TypeError as te:
            raise ValueError(f"ORDER BY failed due to incomparable values: {te}") from te

    # Build SELECT projection
    result: List[Dict[str, Any]] = []
    if select_fields == ["*"]:
        for r in filtered:
            # Shallow copy to avoid mutating original records
            result.append(dict(r))
    else:
        for r in filtered:
            row = {f: r[f] for f in select_fields if f in r}
            result.append(row)

    return result
