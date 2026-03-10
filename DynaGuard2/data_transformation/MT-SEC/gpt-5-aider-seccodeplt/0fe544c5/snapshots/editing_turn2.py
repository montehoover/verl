from typing import Iterable, Mapping, Sequence, List, Dict, Any, Optional, Set

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
