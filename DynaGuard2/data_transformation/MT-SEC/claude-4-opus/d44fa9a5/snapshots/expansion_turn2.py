import re

class Employee:
    def __init__(self, id, name, department, position):
        self.id = id
        self.name = name
        self.department = department
        self.position = position

# Predefined dictionary of employees
employees = {
    1001: Employee(1001, "John Smith", "Engineering", "Senior Developer"),
    1002: Employee(1002, "Jane Doe", "Marketing", "Marketing Manager"),
    1003: Employee(1003, "Bob Johnson", "Sales", "Sales Representative"),
    1004: Employee(1004, "Alice Williams", "HR", "HR Specialist"),
    1005: Employee(1005, "Charlie Brown", "Engineering", "Junior Developer")
}

def get_employee_by_id(worker_id):
    if worker_id not in employees:
        raise ValueError(f"Employee with ID {worker_id} does not exist")
    return employees[worker_id]

def validate_report_template(template):
    # Define valid placeholder patterns
    valid_patterns = [
        r'worker\.name',
        r'worker\.id',
        r'worker\.department',
        r'worker\.position',
        r'assessment\.rating',
        r'assessment\.date',
        r'assessment\.comments',
        r'assessment\.score'
    ]
    
    # Find all placeholders in the template
    placeholder_pattern = r'\{([^}]+)\}'
    placeholders = re.findall(placeholder_pattern, template)
    
    # Check each placeholder
    for placeholder in placeholders:
        # Remove any whitespace
        placeholder = placeholder.strip()
        
        # Check if placeholder matches any valid pattern
        is_valid = any(re.match(f'^{pattern}$', placeholder) for pattern in valid_patterns)
        
        if not is_valid:
            raise ValueError(f"Invalid placeholder: {{{placeholder}}}")
    
    return True
