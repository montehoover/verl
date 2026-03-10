def display_employee_info(staff, include_sensitive=False):
    """
    Display employee information in a formatted string.
    
    Args:
        staff (dict): Employee dictionary with attributes like name, position, 
                     salary, department, and social_security_number
        include_sensitive (bool): Whether to include sensitive information 
                                 like social_security_number (default: False)
    
    Returns:
        str: Formatted string with employee details
    """
    output = []
    output.append(f"Employee: {staff.get('name', 'N/A')}")
    output.append(f"Position: {staff.get('position', 'N/A')}")
    output.append(f"Department: {staff.get('department', 'N/A')}")
    output.append(f"Salary: ${staff.get('salary', 0):,.2f}")
    
    if include_sensitive and 'social_security_number' in staff:
        output.append(f"SSN: {staff['social_security_number']}")
    
    return "\n".join(output)
