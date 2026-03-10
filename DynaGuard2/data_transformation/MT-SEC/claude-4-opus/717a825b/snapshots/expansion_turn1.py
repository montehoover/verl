from collections import defaultdict

def select_fields(records, fields):
    """
    Extract specific fields from a list of dictionaries.
    
    Args:
        records: List of dictionaries representing records
        fields: List of field names (strings) to extract
        
    Returns:
        List of dictionaries containing only the specified fields
    """
    result = []
    
    for record in records:
        selected_record = {}
        for field in fields:
            if field in record:
                selected_record[field] = record[field]
        result.append(selected_record)
    
    return result
