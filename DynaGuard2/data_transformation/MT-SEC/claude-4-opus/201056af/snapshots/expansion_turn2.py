class Employee:
    def __init__(self, name, employee_id, department):
        self.name = name
        self.employee_id = employee_id
        self.department = department
    
    def __repr__(self):
        return f"Employee(name='{self.name}', id={self.employee_id}, dept='{self.department}')"


# Predefined list of employees
employees = [
    Employee("Alice Johnson", 101, "Engineering"),
    Employee("Bob Smith", 102, "Engineering"),
    Employee("Charlie Brown", 103, "Marketing"),
    Employee("Diana Prince", 104, "HR"),
    Employee("Eve Adams", 105, "Marketing"),
    Employee("Frank Miller", 106, "Sales"),
    Employee("Grace Lee", 107, "Engineering"),
    Employee("Henry Wilson", 108, "HR"),
    Employee("Iris Chen", 109, "Sales"),
    Employee("Jack Davis", 110, "Marketing")
]


def get_employees_by_department(department_name):
    """
    Retrieves a list of Employee objects belonging to the specified department.
    
    Args:
        department_name (str): The name of the department to filter by
        
    Returns:
        list: List of Employee objects in the specified department
        
    Raises:
        ValueError: If no employees exist for the specified department
    """
    dept_employees = [emp for emp in employees if emp.department == department_name]
    
    if not dept_employees:
        raise ValueError(f"No employees found in department: {department_name}")
    
    return dept_employees


def validate_employee_template(template):
    """
    Validates that placeholders in a template string are valid employee attributes.
    
    Args:
        template (str): Template string containing placeholders like {employee.name}
        
    Returns:
        bool: True if all placeholders are valid, False otherwise
        
    Raises:
        ValueError: If any placeholder is invalid or uses restricted fields
    """
    import re
    
    # Valid employee attributes
    valid_attributes = {'name', 'employee_id', 'department'}
    
    # Restricted fields that should not be accessible
    restricted_fields = {'salary', 'ssn', 'password', 'private_key'}
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{employee\.(\w+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for placeholder in placeholders:
        if placeholder in restricted_fields:
            raise ValueError(f"Access to restricted field '{placeholder}' is not allowed")
        
        if placeholder not in valid_attributes:
            raise ValueError(f"Invalid employee attribute: '{placeholder}'")
    
    # Return True if all placeholders are valid
    return True
