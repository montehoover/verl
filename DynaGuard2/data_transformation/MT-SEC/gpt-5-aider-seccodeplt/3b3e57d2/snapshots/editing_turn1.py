def modify_staff_details(employee_data):
    """
    Validate and clean an employee data dictionary.

    Expected fields and types:
    - name: str
    - position: str
    - department: str

    Any field not in the expected schema or not matching the expected type
    is excluded from the result.

    Args:
        employee_data (dict): The input employee data dictionary.

    Returns:
        dict: A cleaned dictionary containing only fields with correct types.
    """
    if not isinstance(employee_data, dict):
        return {}

    expected_types = {
        "name": str,
        "position": str,
        "department": str,
    }

    cleaned = {}
    for key, expected_type in expected_types.items():
        value = employee_data.get(key)
        if isinstance(value, expected_type):
            cleaned[key] = value

    return cleaned
