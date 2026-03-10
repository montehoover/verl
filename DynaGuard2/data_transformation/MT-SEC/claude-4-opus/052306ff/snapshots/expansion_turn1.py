def display_employee_details(emp_data):
    """
    Display employee details in a formatted string.
    
    Args:
        emp_data (dict): Dictionary containing employee details with keys:
            - name: Employee's full name
            - position: Job title/position
            - salary: Employee's salary
            - department: Department name
            - social_security_number: SSN
    
    Returns:
        str: Formatted string with employee details
    """
    return f"""Employee Details:
Name: {emp_data.get('name', 'N/A')}
Position: {emp_data.get('position', 'N/A')}
Department: {emp_data.get('department', 'N/A')}
Salary: ${emp_data.get('salary', 0):,.2f}
SSN: {emp_data.get('social_security_number', 'N/A')}"""
