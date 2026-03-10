def extract_fields(data_list, field_names):
    """
    Extract specified fields from a list of dictionaries.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
    
    Returns:
        New list of dictionaries with only the specified fields
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
    Filter records based on conditions and extract specified fields.
    
    Args:
        data_list: List of dictionaries containing the data
        field_names: List of field names to extract
        filter_conditions: Dictionary with field names as keys and 
                          callable conditions or values as values
    
    Returns:
        New list of dictionaries with only the specified fields
        from records that match all filter conditions
    """
    result = []
    for record in data_list:
        # Check if record meets all filter conditions
        meets_conditions = True
        for field, condition in filter_conditions.items():
            if field not in record:
                meets_conditions = False
                break
            
            # If condition is callable, use it as a function
            if callable(condition):
                if not condition(record[field]):
                    meets_conditions = False
                    break
            # Otherwise, check for equality
            else:
                if record[field] != condition:
                    meets_conditions = False
                    break
        
        # If record meets all conditions, extract specified fields
        if meets_conditions:
            extracted_record = {}
            for field in field_names:
                if field in record:
                    extracted_record[field] = record[field]
            result.append(extracted_record)
    
    return result
