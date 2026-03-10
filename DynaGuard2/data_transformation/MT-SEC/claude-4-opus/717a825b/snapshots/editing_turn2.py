def extract_fields(records, fields, filters=None):
    """
    Extract specified fields from a list of dictionaries with optional filtering.
    
    Args:
        records: List of dictionaries
        fields: List of field names to extract
        filters: Dictionary of field names to filter values/functions
                 e.g., {'age': 25, 'name': lambda x: x.startswith('A')}
        
    Returns:
        List of dictionaries containing only the specified fields that meet filter conditions
        
    Raises:
        ValueError: If a field is not found in any record
    """
    result = []
    
    for record in records:
        # Check if record meets filter conditions
        if filters:
            skip_record = False
            for filter_field, filter_value in filters.items():
                if filter_field not in record:
                    raise ValueError(f"Filter field '{filter_field}' not found in record")
                
                # If filter_value is callable, use it as a predicate function
                if callable(filter_value):
                    if not filter_value(record[filter_field]):
                        skip_record = True
                        break
                # Otherwise, check for equality
                else:
                    if record[filter_field] != filter_value:
                        skip_record = True
                        break
            
            if skip_record:
                continue
        
        # Extract specified fields
        extracted = {}
        for field in fields:
            if field not in record:
                raise ValueError(f"Field '{field}' not found in record")
            extracted[field] = record[field]
        result.append(extracted)
    
    return result
