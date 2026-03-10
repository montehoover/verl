from collections import defaultdict

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
