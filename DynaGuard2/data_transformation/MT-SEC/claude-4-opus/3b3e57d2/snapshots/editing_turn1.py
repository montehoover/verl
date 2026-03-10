def modify_staff_details(employee_data):
    expected_types = {
        'name': str,
        'position': str,
        'department': str
    }
    
    cleaned_data = {}
    
    for field, expected_type in expected_types.items():
        if field in employee_data and isinstance(employee_data[field], expected_type):
            cleaned_data[field] = employee_data[field]
    
    return cleaned_data
