def extract_fields(dataset, fields, conditions=None):
    """
    Extract specific fields from a list of dictionaries with optional filtering.
    
    Args:
        dataset: List of dictionaries representing the dataset
        fields: List of field names to extract
        conditions: Optional dictionary of field:value pairs for filtering
        
    Returns:
        List of dictionaries containing only the specified fields that meet conditions
        
    Raises:
        ValueError: If a condition references a non-existent field
    """
    if conditions is None:
        conditions = {}
    
    # Check if all condition fields exist in at least one record
    if conditions and dataset:
        all_fields = set()
        for record in dataset:
            all_fields.update(record.keys())
        
        for condition_field in conditions:
            if condition_field not in all_fields:
                raise ValueError(f"Condition field '{condition_field}' does not exist in dataset")
    
    result = []
    for record in dataset:
        # Check if record meets all conditions
        meets_conditions = True
        for field, value in conditions.items():
            if field not in record or record[field] != value:
                meets_conditions = False
                break
        
        if meets_conditions:
            # Extract only the requested fields
            filtered_record = {field: record.get(field) for field in fields}
            result.append(filtered_record)
    
    return result
