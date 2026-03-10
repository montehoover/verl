def alter_employee_details(emp_data: dict) -> bool:
    """
    Check employee data for consistency.
    
    Args:
        emp_data: Dictionary containing employee information
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    # Check if required fields exist and have correct types
    if 'name' in emp_data and not isinstance(emp_data['name'], str):
        return False
    
    if 'position' in emp_data and not isinstance(emp_data['position'], str):
        return False
    
    if 'salary' in emp_data and not isinstance(emp_data['salary'], (int, float)):
        return False
    
    return True
