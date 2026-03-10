def modify_staff_details(employee_data):
    required_fields = {'name', 'position', 'department'}
    missing_fields = []
    
    for field in required_fields:
        if field not in employee_data:
            missing_fields.append(field)
    
    return missing_fields
