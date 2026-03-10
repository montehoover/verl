import re

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
    Employee("David Brown", "Senior Developer", "Engineering", 92000, "david@personal.com"),
    Employee("Emma Davis", "Marketing Manager", "Marketing", 88000, "emma@personal.com"),
    Employee("Frank Wilson", "HR Specialist", "HR", 70000, "frank@personal.com"),
    Employee("Grace Lee", "Software Engineer", "Engineering", 86000, "grace@personal.com"),
    Employee("Henry Martin", "Sales Representative", "Sales", 65000, "henry@personal.com"),
    Employee("Iris Chen", "HR Manager", "HR", 90000, "iris@personal.com"),
    Employee("Jack Taylor", "Financial Analyst", "Finance", 78000, "jack@personal.com")
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

def create_employee_directory(department, template):
    """
    Generate a formatted directory string for a department using a template.
    
    Args:
        department (str): The department name to generate the directory for
        template (str): The template string with placeholders to be replaced
        
    Returns:
        str: The generated directory string with placeholders replaced
        
    Raises:
        ValueError: If any error occurs or if access to sensitive information is attempted
    """
    try:
        # Validate the template first
        validate_directory_template(template)
        
        # Get employees in the department
        dept_employees = get_employees_by_department(department)
        
        # Build the directory string
        directory_lines = []
        
        for employee in dept_employees:
            # Create a copy of the template for this employee
            employee_entry = template
            
            # Replace placeholders with actual data
            # Note: Some placeholders might not exist as attributes
            replacements = {
                '{employee.name}': employee.name,
                '{employee.position}': employee.position,
                '{employee.department}': employee.department,
                '{employee.employee_id}': str(getattr(employee, 'employee_id', 'N/A')),
                '{employee.email}': getattr(employee, 'email', f"{employee.name.lower().replace(' ', '.')}@company.com"),
                '{employee.phone}': getattr(employee, 'phone', 'N/A'),
                '{employee.office}': getattr(employee, 'office', 'N/A'),
                '{employee.title}': getattr(employee, 'title', employee.position)
            }
            
            for placeholder, value in replacements.items():
                employee_entry = employee_entry.replace(placeholder, value)
            
            directory_lines.append(employee_entry)
        
        return '\n'.join(directory_lines)
        
    except Exception as e:
        raise ValueError(f"Error creating employee directory: {str(e)}")
