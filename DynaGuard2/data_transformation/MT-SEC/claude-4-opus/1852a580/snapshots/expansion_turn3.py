import re

class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email

employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com"),
    Employee("John Smith", "Senior Developer", "Engineering", 90000, "john@personal.com"),
    Employee("Sarah Johnson", "Tech Lead", "Engineering", 100000, "sarah@personal.com"),
    Employee("Michael Brown", "Sales Executive", "Sales", 70000, "michael@personal.com"),
    Employee("Emily Davis", "HR Manager", "HR", 80000, "emily@personal.com"),
    Employee("David Wilson", "Sales Director", "Sales", 110000, "david@personal.com"),
    Employee("Jennifer Lee", "Marketing Specialist", "Marketing", 65000, "jennifer@personal.com"),
    Employee("Robert Taylor", "DevOps Engineer", "Engineering", 88000, "robert@personal.com"),
    Employee("Lisa Anderson", "HR Coordinator", "HR", 60000, "lisa@personal.com"),
    Employee("James Martinez", "Content Manager", "Marketing", 72000, "james@personal.com"),
    Employee("Mary Thomas", "Account Manager", "Sales", 75000, "mary@personal.com")
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
    restricted_placeholders = {'salary', 'ssn', 'private', 'confidential', 'personal_email'}
    
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

def generate_team_roster(division_name, roster_template):
    """
    Generate a formatted employee directory based on a template.
    
    Args:
        division_name (str): The name of the department/division to generate roster for
        roster_template (str): Template string with placeholders like {employee.name}
        
    Returns:
        str: Formatted directory string with employee details
        
    Raises:
        ValueError: If restricted fields are accessed or if an error occurs
    """
    try:
        # Get employees for the specified department
        department_employees = get_employees_by_department(division_name)
        
        # Validate the template
        validate_roster_template(roster_template)
        
        # Build the formatted directory
        roster_lines = []
        
        for employee in department_employees:
            # Create a copy of the template for this employee
            employee_line = roster_template
            
            # Find all placeholders in the template
            placeholder_pattern = r'\{([^}]+)\}'
            placeholders = re.findall(placeholder_pattern, roster_template)
            
            for placeholder in placeholders:
                if placeholder.startswith('employee.'):
                    attribute = placeholder.split('.', 1)[1]
                    
                    # Check for restricted fields
                    restricted_fields = {'salary', 'personal_email', 'ssn', 'private', 'confidential'}
                    if attribute in restricted_fields:
                        raise ValueError(f"Access to restricted field '{attribute}' is not allowed")
                    
                    # Get the attribute value
                    if hasattr(employee, attribute):
                        value = getattr(employee, attribute)
                        employee_line = employee_line.replace(f'{{{placeholder}}}', str(value))
                    else:
                        # Handle employee_id specially since it might not be in the new Employee class
                        if attribute == 'employee_id':
                            employee_line = employee_line.replace(f'{{{placeholder}}}', 'N/A')
                        else:
                            raise ValueError(f"Employee has no attribute '{attribute}'")
            
            roster_lines.append(employee_line)
        
        # Join all employee lines into the final roster
        return '\n'.join(roster_lines)
        
    except ValueError:
        # Re-raise ValueError as is
        raise
    except Exception as e:
        # Wrap any other exceptions in ValueError
        raise ValueError(f"Unexpected error occurred: {str(e)}")
