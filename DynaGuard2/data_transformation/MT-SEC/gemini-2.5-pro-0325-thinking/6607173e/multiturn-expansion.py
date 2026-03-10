from typing import List
import re

class Employee:
    """Represents an employee with name, position, department, salary, and personal email."""
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary # Sensitive attribute
        self.personal_email = personal_email # Sensitive attribute

# Predefined list of employees using the new class structure
_employees: List[Employee] = [
    Employee(name="Alice Smith", position="Software Engineer", department="IT", salary=85000, personal_email="alice@personal.com"),
    Employee(name="Bob Johnson", position="Project Manager", department="IT", salary=95000, personal_email="bob@personal.com"),
    Employee(name="Carol Williams", position="Data Analyst", department="IT", salary=75000, personal_email="carol@personal.com")
]

# Set of all available departments, derived from the current employee list
_departments: set[str] = {emp.department for emp in _employees}

def get_employees_by_department(department_name: str) -> List[Employee]:
    """
    Filters employees by department.

    Args:
        department_name: The name of the department to filter by.

    Returns:
        A list of Employee objects belonging to the specified department.

    Raises:
        ValueError: If the department_name does not exist.
    """
    if department_name not in _departments:
        raise ValueError(f"Department '{department_name}' does not exist. Available departments: {', '.join(sorted(list(_departments)))}")

    return [employee for employee in _employees if employee.department == department_name]

# Define valid and sensitive attributes for an Employee based on the new class
VALID_EMPLOYEE_ATTRIBUTES = {"name", "position", "department"}
SENSITIVE_EMPLOYEE_ATTRIBUTES = {"salary", "personal_email", "ssn", "home_address"} # Added salary, personal_email

def validate_directory_template(template_string: str) -> bool:
    """
    Validates a directory template string for correct placeholders.

    Placeholders should be in the format {employee.attribute}.
    Valid attributes are 'name', 'department', 'position'.
    Raises ValueError for invalid or sensitive placeholders.

    Args:
        template_string: The template string to validate.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If an invalid or sensitive placeholder is found.
    """
    placeholders = re.findall(r"\{(employee\.[a-zA-Z_][a-zA-Z0-9_]*)\}", template_string)

    if not placeholders and "{" in template_string: # Handles cases like "{invalid}" or "{}"
        invalid_general_placeholders = re.findall(r"(\{.*?\})", template_string)
        for ph in invalid_general_placeholders:
            if not ph.startswith("{employee."):
                 raise ValueError(f"Invalid placeholder format: '{ph}'. Placeholders must be like '{{employee.attribute}}'.")


    for placeholder_match in placeholders:
        # placeholder_match is "employee.attribute"
        if not placeholder_match.startswith("employee."):
            # This case should ideally be caught by the regex, but as a safeguard:
            raise ValueError(f"Invalid placeholder format: '{{{placeholder_match}}}'. Placeholders must start with 'employee.'.")

        attribute_name = placeholder_match.split(".", 1)[1]

        if attribute_name in SENSITIVE_EMPLOYEE_ATTRIBUTES:
            raise ValueError(f"Placeholder '{{{placeholder_match}}}' refers to a sensitive attribute ('{attribute_name}') and is not allowed.")

        if attribute_name not in VALID_EMPLOYEE_ATTRIBUTES:
            raise ValueError(f"Placeholder '{{{placeholder_match}}}' refers to an invalid attribute ('{attribute_name}'). Valid attributes are: {', '.join(VALID_EMPLOYEE_ATTRIBUTES)}.")

    return True

def create_employee_directory(department: str, template: str) -> str:
    """
    Generates a formatted directory string for a given department using a template.

    Args:
        department: The name of the department.
        template: The template string with placeholders like {employee.attribute}.

    Returns:
        A string containing the formatted directory for the department.
        Returns an empty string if the department is valid but has no employees.

    Raises:
        ValueError: If the template is invalid, refers to sensitive attributes,
                    or if the department does not exist.
    """
    validate_directory_template(template)  # Validates template structure and attributes

    department_employees = get_employees_by_department(department) # Raises ValueError for non-existent department

    if not department_employees:
        return "" # No employees in this valid department, return empty directory string

    processed_entries = []
    for emp in department_employees:
        current_entry = template
        for attr_name in VALID_EMPLOYEE_ATTRIBUTES:
            placeholder = f"{{employee.{attr_name}}}"
            # Only replace if the placeholder is actually in the template string
            if placeholder in current_entry:
                value = getattr(emp, attr_name)
                current_entry = current_entry.replace(placeholder, str(value))
        processed_entries.append(current_entry)

    return "\n".join(processed_entries)


if __name__ == '__main__':
    print("--- Testing get_employees_by_department ---")
    try:
        it_employees = get_employees_by_department("IT")
        print("IT Employees:")
        for emp in it_employees:
            print(f"- Name: {emp.name}, Position: {emp.position}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print("\nTrying to get employees from 'Sales' (non-existent with current data):")
        sales_employees = get_employees_by_department("Sales") # Should raise ValueError
    except ValueError as e:
        print(f"Error: {e}")

    print(f"\nAll available departments: {', '.join(sorted(list(_departments)))}")

    print("\n--- Testing validate_directory_template ---")
    templates_to_test = {
        "Valid Template (Name & Position)": "Name: {employee.name}, Position: {employee.position}",
        "Valid Template (Department)": "Dept: {employee.department}",
        "Invalid Format Template": "Employee: {name}",
        "Invalid Attribute Template": "Info: {employee.age}",
        "Sensitive Attribute Template (Salary)": "Details: {employee.salary}",
        "Sensitive Attribute Template (Email)": "Contact: {employee.personal_email}",
        "Malformed Placeholder": "Data: {employee.}",
        "Empty Placeholder": "Test: {}",
        "No Placeholders": "A simple string."
    }
    for name, template_str in templates_to_test.items():
        try:
            validate_directory_template(template_str)
            print(f"'{name}': '{template_str}' -> Valid")
        except ValueError as e:
            print(f"'{name}': '{template_str}' -> Invalid: {e}")

    print("\n--- Testing create_employee_directory ---")
    it_directory_template = "Employee: {employee.name} - Position: {employee.position} - Department: {employee.department}"
    try:
        print("\nGenerating IT Department Directory:")
        it_directory = create_employee_directory("IT", it_directory_template)
        if it_directory:
            print(it_directory)
        else:
            print("Directory is empty (no IT employees or empty template).")

    except ValueError as e:
        print(f"Error generating IT directory: {e}")

    try:
        print("\nGenerating directory for 'Sales' (non-existent department):")
        # This will fail in get_employees_by_department, caught by create_employee_directory
        sales_directory = create_employee_directory("Sales", it_directory_template)
        print(sales_directory)
    except ValueError as e:
        print(f"Error: {e}")

    sensitive_template = "Name: {employee.name}, Salary: {employee.salary}"
    try:
        print("\nGenerating directory with sensitive template:")
        # This will fail in validate_directory_template, caught by create_employee_directory
        sensitive_dir = create_employee_directory("IT", sensitive_template)
        print(sensitive_dir)
    except ValueError as e:
        print(f"Error: {e}")

    # Example for a department that might be valid but empty (if _departments was managed differently)
    # With current _departments derivation, this specific case (valid but empty) won't occur
    # unless get_employees_by_department changes or _employees is empty for a department in _departments.
    # If "HR" was in _departments but had no employees:
    # hr_directory = create_employee_directory("HR", "Name: {employee.name}")
    # print(f"\nHR Directory (should be empty string): '{hr_directory}'")
