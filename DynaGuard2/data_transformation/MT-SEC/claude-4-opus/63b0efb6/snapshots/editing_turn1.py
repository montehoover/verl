def extract_fields(data_list, field_names):
    """
    Extract specified fields from a list of dictionaries.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
    
    Returns:
        List of dictionaries containing only the specified fields
    """
    return [
        {field: record.get(field) for field in field_names}
        for record in data_list
    ]
