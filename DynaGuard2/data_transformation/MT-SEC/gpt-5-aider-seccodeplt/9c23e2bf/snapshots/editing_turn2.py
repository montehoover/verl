from typing import Any, Iterable, Mapping, Sequence, List, Dict, Callable, Tuple
from collections.abc import Sequence as _Sequence

__all__ = ["extract_fields", "filter_and_extract"]


def extract_fields(records: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Return a new list of dictionaries containing only the specified fields from each input record.

    - If a requested field is missing from a record, it is omitted from that record's result.
    - Duplicate field names in `fields` are ignored (first occurrence order is preserved).

    Args:
        records: An iterable of dictionaries (mappings) representing the dataset.
        fields: A sequence of field names to extract.

    Returns:
        A list of dictionaries containing only the requested fields for each input record.

    Raises:
        TypeError: If any record is not a mapping or fields is not a sequence of strings.

    Example:
        >>> data = [
        ...     {"id": 1, "name": "Alice", "age": 30},
        ...     {"id": 2, "name": "Bob", "city": "NYC"},
        ... ]
        >>> extract_fields(data, ["id", "name"])
        [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
    """
    # Validate fields are strings and deduplicate while preserving order
    unique_fields: List[str] = []
    for f in fields:
        if not isinstance(f, str):
            raise TypeError("All field names must be strings.")
        if f not in unique_fields:
            unique_fields.append(f)

    result: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            raise TypeError("Each record must be a mapping (e.g., dict).")
        projected = {k: rec[k] for k in unique_fields if k in rec}
        result.append(projected)

    return result


def filter_and_extract(
    records: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
    conditions: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    """
    Filter records by conditions and then project only the requested fields.

    The `conditions` mapping specifies per-field constraints. Each value in `conditions`
    can be one of the following forms:
      - Literal: record[field] must equal the literal value.
      - Callable: a predicate that receives record[field] and returns True/False.
      - Sequence (list/tuple/set) of allowed values: record[field] must be in the sequence.
      - Tuple(operator, operand): supported operators are
            "==", "!=", ">", ">=", "<", "<=",
            "in", "not in", "contains", "icontains",
            "startswith", "endswith"
        Examples:
            ("in", {1, 2, 3}), (">", 10), ("icontains", "abc")
      - Operator dict: a dict of operators to operands (all must match), using keys:
            "$eq", "$ne", "$gt", "$gte", "$lt", "$lte",
            "$in", "$nin", "$contains", "$icontains",
            "$startswith", "$endswith"

    Notes:
      - If a condition references a field that is missing in a record, that record does not match.

    Args:
        records: Iterable of mapping records to filter and project.
        fields: Sequence of field names to include in the output.
        conditions: Mapping of field -> condition (see forms above).

    Returns:
        List of projected dictionaries for records that satisfy all conditions.

    Raises:
        TypeError: If inputs are of incorrect types or unsupported operators are used.

    Example:
        >>> data = [
        ...     {"id": 1, "name": "Alice", "age": 30, "city": "NYC"},
        ...     {"id": 2, "name": "Bob", "age": 25, "city": "LA"},
        ...     {"id": 3, "name": "Carol", "age": 35, "city": "NYC"},
        ... ]
        >>> filter_and_extract(data, ["id", "name"], {"city": "NYC", "age": (">=", 30)})
        [{'id': 1, 'name': 'Alice'}, {'id': 3, 'name': 'Carol'}]
    """
    if not isinstance(conditions, Mapping):
        raise TypeError("conditions must be a mapping of field -> condition")

    # Validate fields are strings and deduplicate while preserving order
    unique_fields: List[str] = []
    for f in fields:
        if not isinstance(f, str):
            raise TypeError("All field names must be strings.")
        if f not in unique_fields:
            unique_fields.append(f)

    def _is_sequence_but_not_str(x: Any) -> bool:
        return isinstance(x, _Sequence) and not isinstance(x, (str, bytes, bytearray))

    def _apply_operator(op: str, value: Any, operand: Any) -> bool:
        if op == "==":
            return value == operand
        if op == "!=":
            return value != operand
        if op == ">":
            return value > operand  # type: ignore[operator]
        if op == ">=":
            return value >= operand  # type: ignore[operator]
        if op == "<":
            return value < operand  # type: ignore[operator]
        if op == "<=":
            return value <= operand  # type: ignore[operator]
        if op == "in":
            return value in operand  # operand should be a container
        if op == "not in":
            return value not in operand
        if op == "contains":
            # record value must contain operand
            try:
                return operand in value
            except TypeError:
                return False
        if op == "icontains":
            # case-insensitive contains; stringify both sides
            try:
                return str(operand).lower() in str(value).lower()
            except Exception:
                return False
        if op == "startswith":
            try:
                return str(value).startswith(str(operand))
            except Exception:
                return False
        if op == "endswith":
            try:
                return str(value).endswith(str(operand))
            except Exception:
                return False
        raise TypeError(f"Unsupported operator: {op}")

    def _evaluate_condition(v: Any, cond: Any) -> bool:
        # Callable predicate
        if callable(cond):
            return bool(cond(v))

        # Operator dict
        if isinstance(cond, Mapping):
            for k, operand in cond.items():
                if k == "$eq":
                    if not _apply_operator("==", v, operand):
                        return False
                elif k == "$ne":
                    if not _apply_operator("!=", v, operand):
                        return False
                elif k == "$gt":
                    if not _apply_operator(">", v, operand):
                        return False
                elif k == "$gte":
                    if not _apply_operator(">=", v, operand):
                        return False
                elif k == "$lt":
                    if not _apply_operator("<", v, operand):
                        return False
                elif k == "$lte":
                    if not _apply_operator("<=", v, operand):
                        return False
                elif k == "$in":
                    if not _apply_operator("in", v, operand):
                        return False
                elif k == "$nin":
                    if not _apply_operator("not in", v, operand):
                        return False
                elif k == "$contains":
                    if not _apply_operator("contains", v, operand):
                        return False
                elif k == "$icontains":
                    if not _apply_operator("icontains", v, operand):
                        return False
                elif k == "$startswith":
                    if not _apply_operator("startswith", v, operand):
                        return False
                elif k == "$endswith":
                    if not _apply_operator("endswith", v, operand):
                        return False
                else:
                    raise TypeError(f"Unsupported operator key in condition dict: {k}")
            return True

        # Tuple operator
        if isinstance(cond, tuple) and len(cond) == 2 and isinstance(cond[0], str):
            op, operand = cond  # type: Tuple[str, Any]
            return _apply_operator(op, v, operand)

        # Sequence of allowed values (membership)
        if _is_sequence_but_not_str(cond):
            try:
                return v in cond  # type: ignore[operator]
            except Exception:
                return False

        # Literal equality
        return v == cond

    result: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, Mapping):
            raise TypeError("Each record must be a mapping (e.g., dict).")

        # Check conditions
        matched = True
        for field, cond in conditions.items():
            if field not in rec:
                matched = False
                break
            if not _evaluate_condition(rec[field], cond):
                matched = False
                break

        if not matched:
            continue

        # Project fields
        projected = {k: rec[k] for k in unique_fields if k in rec}
        result.append(projected)

    return result
