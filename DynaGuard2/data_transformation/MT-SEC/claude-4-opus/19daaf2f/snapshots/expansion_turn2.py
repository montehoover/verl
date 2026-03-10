from collections import defaultdict
from typing import Callable, List, Dict

def select_columns(records, field_names):
    """
    Extract specific fields from a list of records.
    
    Args:
        records: List of dictionaries representing records
        field_names: List of field names (strings) to extract
        
    Returns:
        List of dictionaries containing only the specified fields
    """
    result = []
    
    for record in records:
        selected_record = {}
        for field in field_names:
            if field in record:
                selected_record[field] = record[field]
        result.append(selected_record)
    
    return result

def filter_data(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filter records based on a condition.
    
    Args:
        records: List of dictionaries representing records
        condition: A callable that takes a record and returns True if it meets the condition
        
    Returns:
        List of records that meet the condition
    """
    result = []
    
    for record in records:
        if condition(record):
            result.append(record)
    
    return result
