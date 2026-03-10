def get_employee_details(employee, exclude_sensitive=True):
    """
    Retrieve and format employee details.
    
    Args:
        employee (dict): Dictionary containing employee information with keys:
            - name: Employee's full name
            - position: Job title/position
            - salary: Employee's salary
            - department: Department name
            - social_security_number: SSN (optional in output)
        exclude_sensitive (bool): If True, excludes sensitive information like SSN
    
    Returns:
        str: Formatted string containing employee details
    """
    details = []
    
    # Add basic information
    if 'name' in employee:
        details.append(f"Name: {employee['name']}")
    
    if 'position' in employee:
        details.append(f"Position: {employee['position']}")
    
    if 'department' in employee:
        details.append(f"Department: {employee['department']}")
    
    if 'salary' in employee:
        details.append(f"Salary: ${employee['salary']:,.2f}")
    
    # Add sensitive information only if not excluded
    if not exclude_sensitive and 'social_security_number' in employee:
        details.append(f"SSN: {employee['social_security_number']}")
    
    return '\n'.join(details)
