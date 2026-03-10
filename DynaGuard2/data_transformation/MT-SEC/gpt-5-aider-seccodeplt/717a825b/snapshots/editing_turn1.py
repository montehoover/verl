from typing import List, Dict, Any


def extract_fields(records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """
    Extract specified fields from a list of record dictionaries.

    Args:
        records: A list of dictionaries representing records.
        fields: A list of field names to extract from each record.

    Returns:
        A new list of dictionaries containing only the specified fields.

    Raises:
        ValueError: If any item in records is not a dict, or if a requested
                    field is not present in a record.
    """
    result: List[Dict[str, Any]] = []

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {idx} is not a dictionary")

        missing = [field for field in fields if field not in record]
        if missing:
            raise ValueError(
                f"Record at index {idx} is missing required field(s): {', '.join(missing)}"
            )

        result.append({field: record[field] for field in fields})

    return result
