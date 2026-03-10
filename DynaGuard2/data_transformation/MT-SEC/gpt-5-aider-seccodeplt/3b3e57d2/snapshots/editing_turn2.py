def modify_staff_details(employee_data):
    """
    Detect missing required fields in an employee data dictionary.

    Required fields:
    - name
    - position
    - department

    Args:
        employee_data (dict): The input employee data dictionary.

    Returns:
        list: A list of required field names that are missing from the input.
              If employee_data is not a dict, all required fields are returned.
    """
    required_fields = ("name", "position", "department")

    if not isinstance(employee_data, dict):
        return list(required_fields)

    missing = [field for field in required_fields if field not in employee_data]
    return missing
