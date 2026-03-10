from collections import defaultdict
from typing import List, Dict, Any, Callable

def select_fields(records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts specific fields from a list of dictionaries.

    Args:
        records: A list of dictionaries (representing records).
        fields: A list of field names (strings) to select.

    Returns:
        A list of dictionaries, where each dictionary contains only the
        specified fields from the original record. If a field is not
        present in a record, it is omitted from the new dictionary.
    """
    result_records: List[Dict[str, Any]] = []
    for record in records:
        selected_record: Dict[str, Any] = {}
        for field_name in fields:
            if field_name in record:
                selected_record[field_name] = record[field_name]
        result_records.append(selected_record)
    return result_records


def filter_data(records: List[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Filters a list of records based on a given condition.

    Args:
        records: A list of dictionaries (representing records).
        condition: A callable that takes a record (dictionary) and
                   returns True if the record satisfies the condition,
                   False otherwise.

    Returns:
        A list of records that satisfy the condition.
    """
    filtered_records: List[Dict[str, Any]] = []
    for record in records:
        if condition(record):
            filtered_records.append(record)
    return filtered_records
