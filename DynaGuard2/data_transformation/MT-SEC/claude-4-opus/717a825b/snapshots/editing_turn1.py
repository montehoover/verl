def extract_fields(records, fields):
    """
    Extract specified fields from a list of dictionaries.
    
    Args:
        records: List of dictionaries
        fields: List of field names to extract
        
    Returns:
        List of dictionaries containing only the specified fields
        
    Raises:
        ValueError: If a field is not found in any record
    """
    result = []
    
    for record in records:
        extracted = {}
        for field in fields:
            if field not in record:
                raise ValueError(f"Field '{field}' not found in record")
            extracted[field] = record[field]
        result.append(extracted)
    
    return result
