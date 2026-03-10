def extract_fields(data_list, field_names, conditions=None):
    """
    Extract specific fields from a list of dictionaries with optional filtering.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
        conditions: Dictionary of field:value pairs for filtering (optional)
    
    Returns:
        List of dictionaries containing only the specified fields that meet all conditions
    
    Raises:
        ValueError: If a condition is based on a non-existent field
    """
    if conditions is None:
        conditions = {}
    
    # Check if all condition fields exist in at least one record
    if conditions and data_list:
        all_fields = set()
        for record in data_list:
            all_fields.update(record.keys())
        
        for field in conditions:
            if field not in all_fields:
                raise ValueError(f"Condition field '{field}' does not exist in any record")
    
    result = []
    for record in data_list:
        # Check if record meets all conditions
        meets_conditions = True
        for field, value in conditions.items():
            if field not in record:
                raise ValueError(f"Condition field '{field}' does not exist in record")
            if record[field] != value:
                meets_conditions = False
                break
        
        if meets_conditions:
            extracted_record = {}
            for field in field_names:
                if field in record:
                    extracted_record[field] = record[field]
            result.append(extracted_record)
    
    return result
