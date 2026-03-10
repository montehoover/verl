from dataclasses import dataclass
from typing import List

import re

@dataclass
class Employee:
    """Represents an employee with a name, department, and position."""
    name: str
    department: str
    position: str

# Predefined list of employees
_employees: List[Employee] = [
    Employee(name="Alice Wonderland", department="Engineering", position="Software Engineer"),
    Employee(name="Bob The Builder", department="Engineering", position="Senior Software Engineer"),
    Employee(name="Charlie Brown", department="Marketing", position="Marketing Manager"),
    Employee(name="Diana Prince", department="Sales", position="Sales Representative"),
    Employee(name="Edward Scissorhands", department="HR", position="HR Specialist"),
    Employee(name="Fiona Gallagher", department="Marketing", position="Content Creator"),
]

# Set of all available departments
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

# Define valid and sensitive attributes for an Employee
VALID_EMPLOYEE_ATTRIBUTES = {"name", "department", "position"}
SENSITIVE_EMPLOYEE_ATTRIBUTES = {"salary", "ssn", "home_address"} # Example sensitive attributes

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


if __name__ == '__main__':
    # Example usage:
    try:
        engineering_employees = get_employees_by_department("Engineering")
        print("Engineering Employees:")
        for emp in engineering_employees:
            print(f"- {emp.name}")

        marketing_employees = get_employees_by_department("Marketing")
        print("\nMarketing Employees:")
        for emp in marketing_employees:
            print(f"- {emp.name}")

        # Example of a non-existent department
        print("\nTrying to get employees from a non-existent department:")
        non_existent_employees = get_employees_by_department("Finance")
    except ValueError as e:
        print(e)

    # Example: Get all unique department names
    print(f"\nAll available departments: {', '.join(sorted(list(_departments)))}")

    # Example usage for validate_directory_template:
    print("\nValidating Directory Templates:")
    valid_template_1 = "Employee: {employee.name} - Department: {employee.department}"
    valid_template_2 = "Position: {employee.position}"
    invalid_template_format = "Employee: {name}" # Invalid format
    invalid_template_attr = "Employee: {employee.age}" # Invalid attribute
    sensitive_template_attr = "Employee: {employee.salary}" # Sensitive attribute
    mixed_template = "Name: {employee.name}, Invalid: {employee.address}"
    empty_placeholder = "Test {}"
    malformed_placeholder = "Test {employee.}"

    templates_to_test = {
        "Valid Template 1": valid_template_1,
        "Valid Template 2": valid_template_2,
        "Invalid Format Template": invalid_template_format,
        "Invalid Attribute Template": invalid_template_attr,
        "Sensitive Attribute Template": sensitive_template_attr,
        "Mixed Valid/Invalid Attr Template": mixed_template,
        "Empty Placeholder Template": empty_placeholder,
        "Malformed Placeholder Template": malformed_placeholder,
        "Template with no placeholders": "Just a string.",
        "Template with {employee.name} and {employee.position}": "{employee.name} works as {employee.position}"
    }

    for name, template in templates_to_test.items():
        try:
            validate_directory_template(template)
            print(f"'{name}': '{template}' -> Valid")
        except ValueError as e:
            print(f"'{name}': '{template}' -> Invalid: {e}")
