def extract_fields(data_list, field_names, conditions=None):
    """
    Extract specified fields from a list of dictionaries with optional filtering.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
        conditions: Optional dictionary specifying field conditions for filtering
    
    Returns:
        List of dictionaries containing only the specified fields that match conditions
    
    Raises:
        ValueError: If a condition references a non-existent field
    """
    if conditions is None:
        conditions = {}
    
    # Check if condition fields exist in at least one record
    if conditions and data_list:
        all_fields = set()
        for record in data_list:
            all_fields.update(record.keys())
        
        for condition_field in conditions:
            if condition_field not in all_fields:
                raise ValueError(f"Condition references non-existent field: '{condition_field}'")
    
    # Filter records based on conditions
    filtered_data = []
    for record in data_list:
        match = True
        for field, value in conditions.items():
            if field not in record or record[field] != value:
                match = False
                break
        if match:
            filtered_data.append(record)
    
    # Extract only specified fields
    return [
        {field: record.get(field) for field in field_names}
        for record in filtered_data
    ]
