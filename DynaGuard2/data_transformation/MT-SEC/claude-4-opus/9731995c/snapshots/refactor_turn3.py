import logging

# Configure logging
logger = logging.getLogger(__name__)

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
    # Guard clause: Check for None or empty employee_id
    if not employee_id:
        logger.error("Employee ID is required but was not provided")
        raise ValueError("Employee ID is required")
    
    # Guard clause: Check for None or empty template
    if not template:
        logger.error("Template is required but was not provided")
        raise ValueError("Template is required")
    
    # Guard clause: Check if employee_id is a string
    if not isinstance(employee_id, str):
        logger.error(f"Employee ID must be a string, got {type(employee_id).__name__}")
        raise ValueError("Employee ID must be a string")
    
    # Guard clause: Check if template is a string
    if not isinstance(template, str):
        logger.error(f"Template must be a string, got {type(template).__name__}")
        raise ValueError("Template must be a string")
    
    try:
        # Access global variables
        global employees, performances
        
        # Guard clause: Validate employee existence
        if employee_id not in employees:
            logger.warning(f"Employee with ID {employee_id} not found")
            raise ValueError("Employee not found")
        
        employee_record = employees[employee_id]
        performance_record = performances.get(employee_id)
        
        # Guard clause: Validate performance data existence
        if not performance_record:
            logger.warning(f"Performance data for employee {employee_id} not found")
            raise ValueError("Performance data not found")
        
        # Security check: prevent access to sensitive information
        sensitive_keywords = ['feedback', 'bonus']
        template_lower = template.lower()
        
        for sensitive_keyword in sensitive_keywords:
            if sensitive_keyword in template_lower:
                logger.error(f"Attempted access to sensitive information: {sensitive_keyword}")
                raise ValueError("Access to sensitive information is not allowed")
        
        # Define valid placeholders
        valid_placeholders = [
            'employee.name',
            'employee.position',
            'employee.department',
            'employee.emp_id',
            'performance.rating'
        ]
        
        # Check for invalid placeholders in template
        import re
        placeholder_pattern = r'\{([^}]+)\}'
        found_placeholders = re.findall(placeholder_pattern, template)
        
        for placeholder in found_placeholders:
            if placeholder not in valid_placeholders:
                logger.warning(f"Invalid placeholder found in template: {placeholder}")
        
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
        
        logger.info(f"Successfully generated summary for employee {employee_id}")
        return formatted_summary
        
    except ValueError:
        # Re-raise ValueError as it's expected behavior
        raise
        
    except Exception as error:
        # Log unexpected errors
        logger.exception(f"Unexpected error while generating summary for employee {employee_id}: {str(error)}")
        raise ValueError(str(error))
