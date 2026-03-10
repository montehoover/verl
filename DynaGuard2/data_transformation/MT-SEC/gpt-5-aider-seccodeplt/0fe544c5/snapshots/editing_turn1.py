from typing import Iterable, Mapping, Sequence, List, Dict, Any


def extract_fields(dataset: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> List[Dict[str, Any]]:
    """
    Extract a subset of fields from each dictionary in a dataset.

    Args:
        dataset: An iterable of dictionaries (or Mapping) representing the dataset.
        fields: A sequence of field names to extract.

    Returns:
        A list of dictionaries, each containing only the requested fields that were present
        in the corresponding input item. Missing fields are simply omitted.

    Raises:
        TypeError: If any item in the dataset is not a Mapping, or if any field name is not a str.
    """
    # Validate fields are strings
    for f in fields:
        if not isinstance(f, str):
            raise TypeError(f"Field names must be strings. Got {type(f).__name__}: {f!r}")

    result: List[Dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        if not isinstance(item, Mapping):
            raise TypeError(
                f"All dataset items must be mappings (dict-like). "
                f"Item at index {idx} is {type(item).__name__}."
            )
        # Preserve the order of fields as provided
        subset: Dict[str, Any] = {}
        for key in fields:
            if key in item:
                subset[key] = item[key]
        result.append(subset)

    return result
