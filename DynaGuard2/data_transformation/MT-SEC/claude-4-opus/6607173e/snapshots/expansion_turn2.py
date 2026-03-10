import re

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
    Employee("Carol Williams", 103, "Marketing"),
    Employee("David Brown", 104, "Sales"),
    Employee("Emma Davis", 105, "Marketing"),
    Employee("Frank Wilson", 106, "HR"),
    Employee("Grace Lee", 107, "Engineering"),
    Employee("Henry Martin", 108, "Sales"),
    Employee("Iris Chen", 109, "HR"),
    Employee("Jack Taylor", 110, "Finance")
]

def get_employees_by_department(department_name):
    """
    Filter employees by department name.
    
    Args:
        department_name (str): The name of the department to filter by
        
    Returns:
        list: List of Employee objects in the specified department
        
    Raises:
        ValueError: If the department does not exist
    """
    # Get all unique departments
    all_departments = set(emp.department for emp in employees)
    
    # Check if the department exists
    if department_name not in all_departments:
        raise ValueError(f"Department '{department_name}' does not exist. Available departments: {', '.join(sorted(all_departments))}")
    
    # Filter employees by department
    return [emp for emp in employees if emp.department == department_name]

def validate_directory_template(template):
    """
    Validate that a directory template only contains valid placeholders.
    
    Args:
        template (str): The template string to validate
        
    Returns:
        bool: True if all placeholders are valid, False otherwise
        
    Raises:
        ValueError: If any invalid or sensitive placeholders are found
    """
    # Define valid placeholders
    valid_placeholders = {
        '{employee.name}',
        '{employee.position}',
        '{employee.department}',
        '{employee.employee_id}',
        '{employee.email}',
        '{employee.phone}',
        '{employee.office}',
        '{employee.title}'
    }
    
    # Define sensitive placeholders that should not be allowed
    sensitive_placeholders = {
        '{employee.salary}',
        '{employee.ssn}',
        '{employee.password}',
        '{employee.bank_account}',
        '{employee.address}',
        '{employee.personal_email}',
        '{employee.date_of_birth}',
        '{employee.social_security}'
    }
    
    # Find all placeholders in the template
    placeholder_pattern = re.compile(r'\{employee\.[a-zA-Z_]+\}')
    found_placeholders = set(placeholder_pattern.findall(template))
    
    # Check for sensitive placeholders
    sensitive_found = found_placeholders.intersection(sensitive_placeholders)
    if sensitive_found:
        raise ValueError(f"Template contains sensitive placeholders: {', '.join(sensitive_found)}")
    
    # Check if all placeholders are valid
    invalid_placeholders = found_placeholders - valid_placeholders
    if invalid_placeholders:
        raise ValueError(f"Template contains invalid placeholders: {', '.join(invalid_placeholders)}")
    
    return True
