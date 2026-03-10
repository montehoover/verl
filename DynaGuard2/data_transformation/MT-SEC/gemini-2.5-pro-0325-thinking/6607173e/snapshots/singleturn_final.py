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
    # Added for more comprehensive testing scenarios
    Employee("David Lee", "HR Specialist", "HR", 70000, "david@personal.com") 
]

SENSITIVE_ATTRIBUTES = {"salary", "personal_email"}

def create_employee_directory(department: str, template: str) -> str:
    """
    Generates a directory string for employees in a given department using a template.

    Args:
        department: The name of the department.
        template: The directory template string with placeholders like {employee.name}.

    Returns:
        The generated directory string. Each employee's entry is on a new line.
        Returns an empty string if no employees are found in the department.

    Raises:
        ValueError: If department or template are not strings,
                    if the template attempts to access sensitive information,
                    refers to non-existent attributes, or any other error occurs.
    """
    if not isinstance(department, str):
        raise ValueError("Department name must be a string.")
    if not isinstance(template, str):
        raise ValueError("Template must be a string.")

    # Find all unique attribute names mentioned in placeholders.
    # A placeholder is like {employee.attribute_name}.
    # Example: from "{employee.name} {employee.position}" -> ["name", "position"]
    attributes_in_template = sorted(list(set(re.findall(r'{employee\.(\w+)}', template))))
    
    # Validate all unique attribute names found in the template.
    for attr_name in attributes_in_template:
        if attr_name in SENSITIVE_ATTRIBUTES:
            raise ValueError(f"Access to sensitive attribute '{attr_name}' is not allowed.")
        
        # Check if the attribute is a valid part of the Employee class structure.
        # This is done by checking against a dummy/prototype instance of Employee.
        if not hasattr(Employee("", "", "", 0, ""), attr_name):
             raise ValueError(f"Attribute '{attr_name}' is not a valid employee attribute or is misspelled.")

    # Filter employees by the specified department.
    department_employees = [emp for emp in employees if emp.department == department]

    if not department_employees:
        return ""

    directory_entries = []
    for employee in department_employees:
        current_entry = template
        # Replace placeholders with actual values for the current employee.
        # Iterate over the unique (and validated) attributes found in the template.
        for attr_name in attributes_in_template:
            placeholder_tag = f"{{employee.{attr_name}}}" # e.g., "{employee.name}"
            
            # getattr is safe here because we've validated attr_name.
            value = getattr(employee, attr_name)
            
            # Replace all occurrences of this placeholder_tag in current_entry.
            current_entry = current_entry.replace(placeholder_tag, str(value))
        
        directory_entries.append(current_entry)

    return "\n".join(directory_entries)
