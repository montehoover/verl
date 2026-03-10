class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email
    
    def __repr__(self):
        return f"Employee(name='{self.name}', position='{self.position}', dept='{self.department}')"


# Predefined list of employees
employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com"),
    Employee("Alice Johnson", "Senior Developer", "Engineering", 90000, "alicej@personal.com"),
    Employee("Bob Smith", "DevOps Engineer", "Engineering", 88000, "bobs@personal.com"),
    Employee("Charlie Brown", "Marketing Manager", "Marketing", 82000, "charlie@personal.com"),
    Employee("Diana Prince", "HR Director", "HR", 95000, "diana@personal.com"),
    Employee("Eve Adams", "Content Strategist", "Marketing", 72000, "eve@personal.com"),
    Employee("Frank Miller", "Sales Executive", "Sales", 78000, "frank@personal.com"),
    Employee("Grace Lee", "Full Stack Developer", "Engineering", 86000, "grace@personal.com"),
    Employee("Henry Wilson", "HR Specialist", "HR", 65000, "henry@personal.com"),
    Employee("Iris Chen", "Sales Manager", "Sales", 92000, "iris@personal.com"),
    Employee("Jack Davis", "SEO Specialist", "Marketing", 68000, "jack@personal.com")
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
    valid_attributes = {'name', 'position', 'department'}
    
    # Restricted fields that should not be accessible
    restricted_fields = {'salary', 'ssn', 'password', 'private_key', 'personal_email'}
    
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


def build_worker_list(team_name, list_template):
    """
    Builds a formatted string of employee details for a specified department using a template.
    
    Args:
        team_name (str): The name of the department/team to list employees for
        list_template (str): Template string with placeholders like {employee.name}
        
    Returns:
        str: Formatted string with all employees in the department
        
    Raises:
        ValueError: If invalid access attempt occurs or placeholder references restricted data
    """
    import re
    
    # First validate the template
    validate_employee_template(list_template)
    
    # Get employees for the specified department
    try:
        team_employees = get_employees_by_department(team_name)
    except ValueError:
        return f"No employees found in department: {team_name}"
    
    # Build the formatted list
    formatted_entries = []
    
    for employee in team_employees:
        # Create a copy of the template for this employee
        entry = list_template
        
        # Find all placeholders and replace them
        placeholder_pattern = r'\{employee\.(\w+)\}'
        placeholders = re.findall(placeholder_pattern, entry)
        
        for placeholder in placeholders:
            # Get the attribute value
            if hasattr(employee, placeholder):
                value = getattr(employee, placeholder)
                entry = entry.replace(f'{{employee.{placeholder}}}', str(value))
        
        formatted_entries.append(entry)
    
    # Join all entries with newlines
    return '\n'.join(formatted_entries)
