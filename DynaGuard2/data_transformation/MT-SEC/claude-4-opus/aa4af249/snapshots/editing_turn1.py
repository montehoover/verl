def check_fields(dataset, field_names):
    """
    Check if all specified field names exist in the dataset.
    
    Args:
        dataset: List of dictionaries representing the dataset
        field_names: List of field names to check
        
    Returns:
        Boolean indicating whether all fields exist in at least one record
    """
    if not dataset or not field_names:
        return False
    
    # Collect all unique keys from the dataset
    all_keys = set()
    for record in dataset:
        all_keys.update(record.keys())
    
    # Check if all requested fields exist
    return all(field in all_keys for field in field_names)
