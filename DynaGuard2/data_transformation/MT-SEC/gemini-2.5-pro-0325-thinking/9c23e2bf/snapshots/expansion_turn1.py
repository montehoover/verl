from collections import defaultdict
from typing import List, Dict, Any, Optional

def select_fields(data_records: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Optional[Any]]]:
    """
    Extracts specific fields from a list of dictionaries.

    Args:
        data_records: A list of dictionaries, where keys are strings and values can be of any type.
        fields: A list of field names (strings) to select.

    Returns:
        A list of dictionaries, where each dictionary contains only the
        specified fields. If a field is not present in an original record,
        it will be included in the corresponding new record with a value of None.
    """
    selected_data: List[Dict[str, Optional[Any]]] = []
    for record in data_records:
        new_record: Dict[str, Optional[Any]] = {}
        for field in fields:
            new_record[field] = record.get(field, None)  # Handles missing fields gracefully with None
        selected_data.append(new_record)
    return selected_data
