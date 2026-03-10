def generate_employee_summary(employee_id: str, template: str) -> str:
    """
    Generate a performance summary for an employee based on a template.
    
    This function creates a customized performance summary by replacing placeholders
    in a template string with actual employee and performance data. It prevents access
    to sensitive information such as feedback and bonus details.
    
    Args:
        employee_id (str): The unique identifier of the employee (e.g., "E101")
        template (str): A template string containing placeholders in the format
                       {placeholder_name}. Valid placeholders include:
                       - {employee.name}: Employee's full name
                       - {employee.position}: Employee's job position
                       - {employee.department}: Employee's department
                       - {employee.emp_id}: Employee's ID
                       - {performance.rating}: Employee's performance rating
    
    Returns:
        str: The generated summary with all valid placeholders replaced
    
    Raises:
        ValueError: If the employee ID is not found, performance data is missing,
                   or if the template attempts to access sensitive information
                   (feedback or bonus details)
    
    Example:
        >>> template = "Employee {employee.name} ({employee.position}) has a rating of {performance.rating}"
        >>> generate_employee_summary("E101", template)
        "Employee John Doe (Senior Software Engineer) has a rating of 4.3"
    """
    try:
        # Access global variables
        global employees, performances
        
        # Validate employee existence
        if employee_id not in employees:
            raise ValueError("Employee not found")
        
        employee_record = employees[employee_id]
        performance_record = performances.get(employee_id)
        
        # Validate performance data existence
        if not performance_record:
            raise ValueError("Performance data not found")
        
        # Security check: prevent access to sensitive information
        sensitive_keywords = ['feedback', 'bonus']
        template_lower = template.lower()
        
        for sensitive_keyword in sensitive_keywords:
            if sensitive_keyword in template_lower:
                raise ValueError("Access to sensitive information is not allowed")
        
        # Build replacement context with allowed data
        replacement_context = {
            'employee.name': employee_record.name,
            'employee.position': employee_record.position,
            'employee.department': employee_record.department,
            'employee.emp_id': employee_record.emp_id,
            'performance.rating': str(performance_record.rating)
        }
        
        # Perform template substitution
        formatted_summary = template
        
        for placeholder_key, placeholder_value in replacement_context.items():
            formatted_summary = formatted_summary.replace(
                f'{{{placeholder_key}}}', 
                placeholder_value
            )
        
        return formatted_summary
        
    except Exception as error:
        raise ValueError(str(error))
