def modify_staff_details(employee_data):
    """
    Detects missing required fields in employee data.

    Args:
        employee_data (dict): A dictionary containing employee details.

    Returns:
        list: A list of fields that are missing from the required set:
              {'name', 'position', 'department'}.
    """
    required_fields = {'name', 'position', 'department'}
    missing_fields = []
    for field in required_fields:
        if field not in employee_data:
            missing_fields.append(field)
    return missing_fields
