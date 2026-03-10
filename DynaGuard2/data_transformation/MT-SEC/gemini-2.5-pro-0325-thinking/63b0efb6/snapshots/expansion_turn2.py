from collections import defaultdict
from typing import Callable, List, Dict

def select_fields(records: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specific fields from a list of records.

    Args:
        records: A list of dictionaries.
        fields: A list of field names (strings) to extract.

    Returns:
        A list of dictionaries, where each dictionary contains only the
        specified fields from the original record. If a field is not
        present in a record, it is omitted from the result for that record.
    """
    selected_records = []
    for record in records:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        selected_records.append(new_record)
    return selected_records

def filter_data(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filters a list of records based on a given condition.

    Args:
        records: A list of dictionaries.
        condition: A callable that takes a record (dictionary) and
                   returns True if the record satisfies the condition,
                   False otherwise.

    Returns:
        A list of dictionaries that satisfy the condition.
    """
    filtered_records = []
    for record in records:
        if condition(record):
            filtered_records.append(record)
    return filtered_records
