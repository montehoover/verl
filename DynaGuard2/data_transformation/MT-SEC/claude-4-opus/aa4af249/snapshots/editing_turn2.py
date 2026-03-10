def check_fields(dataset, field_names, conditions=None):
    """
    Check if all specified field names exist in the dataset and optionally if any records match conditions.
    
    Args:
        dataset: List of dictionaries representing the dataset
        field_names: List of field names to check
        conditions: Optional dictionary specifying field-value pairs to match
        
    Returns:
        Boolean indicating whether all fields exist and conditions are met (if provided)
    """
    if not dataset or not field_names:
        return False
    
    # Collect all unique keys from the dataset
    all_keys = set()
    for record in dataset:
        all_keys.update(record.keys())
    
    # Check if all requested fields exist
    fields_exist = all(field in all_keys for field in field_names)
    
    if not fields_exist:
        return False
    
    # If no conditions specified, return True since fields exist
    if conditions is None:
        return True
    
    # Check if any record matches all conditions
    for record in dataset:
        if all(record.get(field) == value for field, value in conditions.items()):
            return True
    
    return False
