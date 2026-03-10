def display_employee_profile(worker):
    """
    Display a formatted employee profile.
    
    Args:
        worker (dict): Dictionary containing employee information with keys:
                      - name (str): Employee's name
                      - position (str): Employee's job position
                      - salary (float/int): Employee's salary
                      - department (str): Employee's department
                      - social_security_number (str): Employee's SSN
    
    Returns:
        str: Formatted string displaying employee profile
    """
    profile = f"""Employee Profile
===================
Name: {worker['name']}
Position: {worker['position']}
Department: {worker['department']}
Salary: ${worker['salary']:,.2f}
SSN: {worker['social_security_number']}"""
    
    return profile
