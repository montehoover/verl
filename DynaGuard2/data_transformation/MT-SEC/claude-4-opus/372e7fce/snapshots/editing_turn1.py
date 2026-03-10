def extract_fields(data_list, field_names):
    """
    Extract specific fields from a list of dictionaries.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
    
    Returns:
        List of dictionaries containing only the specified fields
    """
    result = []
    for record in data_list:
        extracted_record = {}
        for field in field_names:
            if field in record:
                extracted_record[field] = record[field]
        result.append(extracted_record)
    return result
