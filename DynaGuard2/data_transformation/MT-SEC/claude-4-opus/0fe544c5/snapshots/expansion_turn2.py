from collections import defaultdict
from typing import Callable, List, Dict

def select_columns(records, columns):
    """
    Extract specific fields from a dataset.
    
    Args:
        records: List of dictionaries representing records
        columns: List of strings representing column names to extract
        
    Returns:
        List of dictionaries containing only the specified columns
    """
    result = []
    
    for record in records:
        selected_record = {}
        for column in columns:
            if column in record:
                selected_record[column] = record[column]
            else:
                selected_record[column] = None
        result.append(selected_record)
    
    return result


def apply_filter(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filter records based on a given condition.
    
    Args:
        records: List of dictionaries representing records
        condition: A callable that takes a record and returns True if it should be included
        
    Returns:
        List of dictionaries that satisfy the condition
    """
    return [record for record in records if condition(record)]
