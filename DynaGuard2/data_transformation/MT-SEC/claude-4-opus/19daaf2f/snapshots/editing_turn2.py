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


def filter_and_extract(data_list, field_names, filter_conditions):
    """
    Filter records based on conditions and extract specific fields.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
        filter_conditions: Dictionary where keys are field names and values are required values
        
    Returns:
        List of dictionaries containing only the specified fields from filtered records
    """
    result = []
    for record in data_list:
        # Check if record matches all filter conditions
        matches = True
        for field, value in filter_conditions.items():
            if field not in record or record[field] != value:
                matches = False
                break
        
        if matches:
            # Extract specified fields
            extracted_record = {}
            for field in field_names:
                if field in record:
                    extracted_record[field] = record[field]
            result.append(extracted_record)
    
    return result
