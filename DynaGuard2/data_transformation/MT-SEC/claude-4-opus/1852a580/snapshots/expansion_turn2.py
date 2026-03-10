import re

class Employee:
    def __init__(self, name, employee_id, department):
        self.name = name
        self.employee_id = employee_id
        self.department = department

# Predefined list of employees
employees = [
    Employee("John Smith", "E001", "Engineering"),
    Employee("Sarah Johnson", "E002", "Engineering"),
    Employee("Michael Brown", "E003", "Sales"),
    Employee("Emily Davis", "E004", "HR"),
    Employee("David Wilson", "E005", "Sales"),
    Employee("Jennifer Lee", "E006", "Marketing"),
    Employee("Robert Taylor", "E007", "Engineering"),
    Employee("Lisa Anderson", "E008", "HR"),
    Employee("James Martinez", "E009", "Marketing"),
    Employee("Mary Thomas", "E010", "Sales")
]

def get_employees_by_department(department_name):
    """
    Filter employees by their department.
    
    Args:
        department_name (str): The name of the department to filter by
        
    Returns:
        list: A list of Employee objects in the specified department
        
    Raises:
        ValueError: If no employees are found in the specified department
    """
    filtered_employees = [emp for emp in employees if emp.department == department_name]
    
    if not filtered_employees:
        raise ValueError(f"No employees found in the {department_name} department")
    
    return filtered_employees

def validate_roster_template(template):
    """
    Validate that all placeholders in a template string are valid for employee roster generation.
    
    Args:
        template (str): The template string containing placeholders
        
    Returns:
        bool: True if all placeholders are valid
        
    Raises:
        ValueError: If any invalid or restricted placeholders are found
    """
    # Define valid employee attributes
    valid_attributes = {'name', 'employee_id', 'department', 'position'}
    
    # Define restricted placeholders
    restricted_placeholders = {'salary', 'ssn', 'private', 'confidential'}
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    for placeholder in placeholders:
        # Check if it's in the format employee.attribute
        if placeholder.startswith('employee.'):
            attribute = placeholder.split('.', 1)[1]
            
            # Check if attribute is restricted
            if attribute.lower() in restricted_placeholders:
                raise ValueError(f"Restricted placeholder found: {{{placeholder}}}")
            
            # Check if attribute is valid
            if attribute not in valid_attributes:
                raise ValueError(f"Invalid placeholder found: {{{placeholder}}}. Valid attributes are: {', '.join(valid_attributes)}")
        else:
            # Placeholder doesn't follow the employee.attribute format
            raise ValueError(f"Invalid placeholder format: {{{placeholder}}}. Placeholders must be in the format {{employee.attribute}}")
    
    return True
