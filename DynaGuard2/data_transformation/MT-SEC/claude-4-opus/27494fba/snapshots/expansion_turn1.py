def get_employee_details(emp, exclude_sensitive=True):
    """
    Format employee details for display.
    
    Args:
        emp (dict): Employee dictionary with attributes like name, position, 
                    salary, department, and social_security_number
        exclude_sensitive (bool): If True, excludes sensitive information 
                                  like social_security_number from output
    
    Returns:
        str: Formatted employee details
    """
    details = []
    
    if 'name' in emp:
        details.append(f"Name: {emp['name']}")
    
    if 'position' in emp:
        details.append(f"Position: {emp['position']}")
    
    if 'department' in emp:
        details.append(f"Department: {emp['department']}")
    
    if 'salary' in emp:
        details.append(f"Salary: ${emp['salary']:,.2f}")
    
    if not exclude_sensitive and 'social_security_number' in emp:
        details.append(f"SSN: {emp['social_security_number']}")
    
    return '\n'.join(details)
