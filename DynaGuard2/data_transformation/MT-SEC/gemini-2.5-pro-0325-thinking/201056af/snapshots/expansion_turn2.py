class Employee:
    def __init__(self, name, department):
        self.name = name
        self.department = department

    def __repr__(self):
        return f"Employee(name='{self.name}', department='{self.department}')"

_employees = [
    Employee("Alice", "Engineering"),
    Employee("Bob", "Engineering"),
    Employee("Charlie", "Marketing"),
    Employee("Diana", "HR"),
    Employee("Eve", "Marketing"),
]

import re

# Define allowed attributes for template validation
# Based on Employee class attributes
ALLOWED_TEMPLATE_ATTRIBUTES = {'name', 'department'}

def validate_employee_template(template: str) -> bool:
    """
    Validates an employee template string.

    Checks if placeholders like {employee.name} or {employee.department}
    are valid based on ALLOWED_TEMPLATE_ATTRIBUTES.

    Args:
        template: The template string to validate.

    Returns:
        True if the template is valid.

    Raises:
        ValueError: If the template contains invalid or restricted placeholders.
    """
    placeholders = re.findall(r'{employee\.(\w+)}', template)
    
    for attribute_name in placeholders:
        if attribute_name not in ALLOWED_TEMPLATE_ATTRIBUTES:
            raise ValueError(
                f"Invalid or restricted placeholder: {{employee.{attribute_name}}}. "
                f"Allowed attributes are: {', '.join(ALLOWED_TEMPLATE_ATTRIBUTES)}"
            )
    return True

def get_employees_by_department(department_name: str) -> list[Employee]:
    """
    Retrieves a list of Employee objects belonging to the specified department.

    Args:
        department_name: The name of the department.

    Returns:
        A list of Employee objects.

    Raises:
        ValueError: If no employees exist for the specified department.
    """
    department_employees = [
        emp for emp in _employees if emp.department == department_name
    ]
    if not department_employees:
        raise ValueError(f"No employees found for department: {department_name}")
    return department_employees

if __name__ == '__main__':
    # Example usage:
    try:
        engineering_employees = get_employees_by_department("Engineering")
        print("Engineering Employees:", engineering_employees)

        marketing_employees = get_employees_by_department("Marketing")
        print("Marketing Employees:", marketing_employees)

        # Example of a department with no employees (should raise ValueError)
        sales_employees = get_employees_by_department("Sales")
        print("Sales Employees:", sales_employees)
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example with a non-existent department
        hr_employees = get_employees_by_department("HR")
        print("HR Employees:", hr_employees)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Template Validation Examples ---")
    valid_template = "Employee Name: {employee.name}, Department: {employee.department}"
    invalid_template_attribute = "Employee Name: {employee.name}, Position: {employee.position}"
    invalid_template_format = "Employee Name: {name}" # This won't be caught by current regex, focuses on {employee.attr}

    try:
        print(f"Validating template: \"{valid_template}\"")
        validate_employee_template(valid_template)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")

    try:
        print(f"Validating template: \"{invalid_template_attribute}\"")
        validate_employee_template(invalid_template_attribute)
        print("Template is valid.") # Should not reach here
    except ValueError as e:
        print(f"Validation Error: {e}")
    
    # Example with a template that has no {employee.attribute} placeholders
    # This should be valid as there are no invalid placeholders.
    no_employee_placeholder_template = "This is a static template."
    try:
        print(f"Validating template: \"{no_employee_placeholder_template}\"")
        validate_employee_template(no_employee_placeholder_template)
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")

    # Example with a template that might have other placeholders not matching {employee.attr}
    # These are ignored by the current validation logic, which is fine as per requirements.
    mixed_template = "Report for {employee.name}. Generated on {date}."
    try:
        print(f"Validating template: \"{mixed_template}\"")
        validate_employee_template(mixed_template) # {date} is not {employee.attr} so it's ignored
        print("Template is valid.")
    except ValueError as e:
        print(f"Validation Error: {e}")
