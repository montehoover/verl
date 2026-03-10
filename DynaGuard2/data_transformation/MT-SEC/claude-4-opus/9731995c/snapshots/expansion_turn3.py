import re

class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, employee, rating, feedback, bonus):
        self.employee = employee
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "E101": Employee("E101", "John Doe", "Senior Software Engineer", "R&D"),
    "E102": Employee("E102", "Jane Smith", "Business Analyst", "Operations"),
}

performances = {
    "E101": Performance(employees["E101"], 4.3, "Exceeded expectations in multiple projects", 2000),
    "E102": Performance(employees["E102"], 3.8, "Consistently met expectations.", 1500),
}

# Legacy performance data (kept for backward compatibility)
performance_data = {
    101: Performance(None, 4.5, "Excellent work on project deliverables", None),
    102: Performance(None, 3.8, "Good teamwork and communication skills", None),
    103: Performance(None, 4.2, "Strong technical skills and problem-solving", None),
    104: Performance(None, 3.5, "Meets expectations, room for improvement", None),
    105: Performance(None, 4.8, "Outstanding leadership and innovation", None)
}

def get_performance_by_id(employee_id):
    """
    Retrieves the Performance object for a given employee ID.
    
    Args:
        employee_id: The ID of the employee
        
    Returns:
        Performance object associated with the employee ID
        
    Raises:
        ValueError: If the employee ID doesn't exist in the performance data
    """
    if employee_id not in performance_data:
        raise ValueError(f"Employee ID {employee_id} not found in performance data")
    
    return performance_data[employee_id]

def check_summary_placeholders(template):
    """
    Verifies that a template string only contains valid placeholders.
    
    Args:
        template: The template string to validate
        
    Returns:
        bool: True if all placeholders are valid, False otherwise
        
    Raises:
        ValueError: If any invalid or sensitive placeholders are found
    """
    # Define valid placeholder patterns
    valid_placeholders = {
        'employee.name',
        'employee.id',
        'employee.department',
        'employee.title',
        'performance.rating',
        'performance.review',
        'performance.employee_id'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    found_placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for placeholder in found_placeholders:
        # Remove any whitespace
        placeholder = placeholder.strip()
        
        # Check if it's a valid placeholder
        if placeholder not in valid_placeholders:
            # Check for potentially sensitive placeholders
            sensitive_patterns = ['password', 'ssn', 'salary', 'secret', 'private', 'confidential']
            if any(pattern in placeholder.lower() for pattern in sensitive_patterns):
                raise ValueError(f"Sensitive placeholder detected: {{{placeholder}}}")
            else:
                raise ValueError(f"Invalid placeholder: {{{placeholder}}}")
    
    return True

def generate_employee_summary(employee_id, template):
    """
    Generates a formatted summary string for an employee using a template.
    
    Args:
        employee_id: The ID of the employee
        template: The template string with placeholders
        
    Returns:
        str: The generated summary with placeholders replaced
        
    Raises:
        ValueError: If the employee ID doesn't exist or if sensitive data is requested
    """
    # Check if employee exists
    if employee_id not in employees:
        raise ValueError(f"Employee ID {employee_id} not found")
    
    # Check if performance data exists
    if employee_id not in performances:
        raise ValueError(f"Performance data for employee ID {employee_id} not found")
    
    employee = employees[employee_id]
    performance = performances[employee_id]
    
    # Check for sensitive placeholders in template
    sensitive_patterns = [
        'feedback',
        'bonus',
        'salary',
        'compensation',
        'review_details',
        'private',
        'confidential'
    ]
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    found_placeholders = re.findall(placeholder_pattern, template)
    
    for placeholder in found_placeholders:
        placeholder_lower = placeholder.strip().lower()
        if any(pattern in placeholder_lower for pattern in sensitive_patterns):
            raise ValueError(f"Access to sensitive data '{placeholder}' is not allowed")
    
    # Replace placeholders with actual data
    result = template
    
    # Employee placeholders
    result = result.replace('{employee.name}', employee.name)
    result = result.replace('{employee.emp_id}', employee.emp_id)
    result = result.replace('{employee.position}', employee.position)
    result = result.replace('{employee.department}', employee.department)
    
    # Performance placeholders
    result = result.replace('{performance.rating}', str(performance.rating))
    result = result.replace('{performance.employee.name}', performance.employee.name)
    result = result.replace('{performance.employee.emp_id}', performance.employee.emp_id)
    result = result.replace('{performance.employee.position}', performance.employee.position)
    result = result.replace('{performance.employee.department}', performance.employee.department)
    
    return result
