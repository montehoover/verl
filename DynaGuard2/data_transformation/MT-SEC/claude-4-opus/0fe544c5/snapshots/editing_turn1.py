def extract_fields(dataset, fields):
    """
    Extract specific fields from a list of dictionaries.
    
    Args:
        dataset: List of dictionaries representing the dataset
        fields: List of field names to extract
        
    Returns:
        List of dictionaries containing only the specified fields
    """
    return [
        {field: record.get(field) for field in fields}
        for record in dataset
    ]
