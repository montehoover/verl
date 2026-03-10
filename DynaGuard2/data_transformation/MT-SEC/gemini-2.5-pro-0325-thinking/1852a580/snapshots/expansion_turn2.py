import re
from typing import List, Set

class Employee:
    """Represents an employee with a name and department."""
    def __init__(self, name: str, department: str):
        self.name = name
        self.department = department

    def __repr__(self) -> str:
        return f"Employee(name='{self.name}', department='{self.department}')"

# Predefined list of employees
_employees: List[Employee] = [
    Employee("Alice", "Engineering"),
    Employee("Bob", "Engineering"),
    Employee("Charlie", "Marketing"),
    Employee("David", "HR"),
    Employee("Eve", "Marketing"),
]

def get_employees_by_department(department_name: str) -> List[Employee]:
    """
    Filters employees by their department.

    Args:
        department_name: The name of the department to filter by.

    Returns:
        A list of Employee objects belonging to the specified department.

    Raises:
        ValueError: If no employees are found in the specified department.
    """
    filtered_employees = [
        emp for emp in _employees if emp.department == department_name
    ]

    if not filtered_employees:
        raise ValueError(f"No employees found in department: {department_name}")

    return filtered_employees

VALID_EMPLOYEE_ATTRIBUTES: Set[str] = {"name", "department"}

def validate_roster_template(template_string: str) -> bool:
    """
    Validates a roster template string for correct employee placeholders.

    Args:
        template_string: The template string to validate.
                         Placeholders should be in the format {employee.attribute}.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any invalid or restricted placeholders are found.
    """
    # Find all placeholders like {employee.attribute_name}
    placeholders = re.findall(r'{employee\.(\w+)}', template_string)

    if not placeholders and '{' in template_string: # Handles cases like {employee} or {employee.}
        # Check for malformed placeholders if any curly brace is present but no valid pattern was matched
        if re.search(r'{employee[^\w\s\.}]*?}', template_string) or re.search(r'{employee\.[^\w\s]*?}', template_string):
             raise ValueError("Malformed employee placeholder found.")


    for attribute_name in placeholders:
        if attribute_name not in VALID_EMPLOYEE_ATTRIBUTES:
            raise ValueError(
                f"Invalid placeholder attribute: '{attribute_name}'. "
                f"Allowed attributes are: {', '.join(VALID_EMPLOYEE_ATTRIBUTES)}."
            )
    return True

if __name__ == '__main__':
    # Example usage for get_employees_by_department:
    try:
        engineering_employees = get_employees_by_department("Engineering")
        print("Engineering Employees:", engineering_employees)

        marketing_employees = get_employees_by_department("Marketing")
        print("Marketing Employees:", marketing_employees)

        # Example of a department with no employees
        sales_employees = get_employees_by_department("Sales")
        print("Sales Employees:", sales_employees)
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example of a department that exists but has no employees in this list
        # (assuming "Finance" is a valid department but no one is assigned yet)
        finance_employees = get_employees_by_department("Finance")
        print("Finance Employees:", finance_employees)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Validating Roster Templates ---")
    # Example usage for validate_roster_template:
    valid_template_1 = "Employee: {employee.name}, Department: {employee.department}"
    valid_template_2 = "Name: {employee.name}"
    invalid_template_1 = "Employee: {employee.name}, Position: {employee.position}"
    invalid_template_2 = "Contact: {employee.phone}"
    empty_template = "No placeholders here."
    malformed_template_1 = "Employee: {employee.}"
    malformed_template_2 = "Employee: {employee}"


    templates_to_test = {
        "Valid Template 1": valid_template_1,
        "Valid Template 2": valid_template_2,
        "Empty Template": empty_template,
        "Invalid Attribute 'position'": invalid_template_1,
        "Invalid Attribute 'phone'": invalid_template_2,
        "Malformed Placeholder (dot only)": malformed_template_1,
        "Malformed Placeholder (no attribute)": malformed_template_2,
    }

    for description, template in templates_to_test.items():
        try:
            print(f"\nTesting template: \"{template}\" ({description})")
            is_valid = validate_roster_template(template)
            print(f"Validation result: {'Valid' if is_valid else 'Invalid (should have raised error)'}")
        except ValueError as e:
            print(f"Validation Error: {e}")
