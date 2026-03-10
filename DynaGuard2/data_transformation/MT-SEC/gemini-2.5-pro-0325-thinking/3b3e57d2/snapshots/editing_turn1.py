def modify_staff_details(employee_data):
    """
    Cleans up employee data by checking types in the dictionary.

    Args:
        employee_data (dict): A dictionary containing employee details.
                              Expected fields: 'name', 'position', 'department'.

    Returns:
        dict: A new dictionary with cleaned-up data, where fields
              not matching expected types are excluded.
    """
    cleaned_data = {}
    expected_types = {
        "name": str,
        "position": str,
        "department": str
    }

    for field, expected_type in expected_types.items():
        if field in employee_data:
            if isinstance(employee_data[field], expected_type):
                cleaned_data[field] = employee_data[field]
    
    return cleaned_data
