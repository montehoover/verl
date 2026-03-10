import re

class Employee:
    def __init__(self, name, employee_id, department, position):
        self.name = name
        self.employee_id = employee_id
        self.department = department
        self.position = position

    def __repr__(self):
        return f"Employee(name='{self.name}', id='{self.employee_id}', department='{self.department}', position='{self.position}')"

# Predefined list of employees
employees = [
    Employee("Alice Smith", "E1001", "Engineering", "Software Engineer"),
    Employee("Bob Johnson", "E1002", "Engineering", "Senior Software Engineer"),
    Employee("Charlie Brown", "E1003", "HR", "HR Manager"),
    Employee("Diana Prince", "E1004", "Marketing", "Marketing Specialist"),
    Employee("Edward Nigma", "E1005", "Engineering", "QA Engineer"),
    Employee("Fiona Glenanne", "E1006", "HR", "HR Assistant"),
]

def list_employees_by_department(department_name):
    """
    Fetches employees from a specific department.

    Args:
        department_name (str): The name of the department.

    Returns:
        list: A list of Employee objects belonging to the specified department.
    """
    return [employee for employee in employees if employee.department == department_name]

# Define approved and sensitive attributes for template validation
APPROVED_EMPLOYEE_ATTRIBUTES = {
    "name",
    "employee_id",
    "department",
    "position",
}

SENSITIVE_EMPLOYEE_ATTRIBUTES = {
    "salary",
    "personal_email",
}

def validate_format_template(template_string):
    """
    Checks if a given string template contains only approved employee placeholders
    and no sensitive employee placeholders.

    Approved placeholders are of the form {employee.attribute} where attribute
    is one of name, employee_id, department, position.
    Placeholders for sensitive attributes like {employee.salary} or
    {employee.personal_email} will cause a ValueError.
    Other non-approved placeholders or malformed placeholders will cause the
    function to return False.

    Args:
        template_string (str): The template string to validate.

    Returns:
        bool: True if the template is compliant (contains only approved placeholders
              and no sensitive ones), False if it contains non-approved or
              malformed placeholders.

    Raises:
        ValueError: If the template contains sensitive placeholders.
    """
    # Regex to find all content within curly braces {}
    placeholders_in_template = re.findall(r"\{([^}]+)\}", template_string)

    if not placeholders_in_template:
        return True # No placeholders, so it's compliant by default

    for content in placeholders_in_template:
        parts = content.split(".", 1)
        
        # Check if placeholder is in the format "employee.attribute"
        # and attribute part is not empty.
        if len(parts) != 2 or parts[0] != "employee" or not parts[1]:
            # Not a valid "employee.attribute" placeholder (e.g., "{foo}", "{employee.}", "{}")
            return False 

        attribute = parts[1]

        if attribute in SENSITIVE_EMPLOYEE_ATTRIBUTES:
            raise ValueError(f"Template contains sensitive placeholder: {{{content}}}")

        if attribute not in APPROVED_EMPLOYEE_ATTRIBUTES:
            # Attribute is not sensitive, but also not in the approved list
            return False
            
    return True # All placeholders were valid, approved, and not sensitive

if __name__ == '__main__':
    engineering_employees = list_employees_by_department("Engineering")
    print("Engineering Department Employees:")
    for emp in engineering_employees:
        print(emp)

    hr_employees = list_employees_by_department("HR")
    print("\nHR Department Employees:")
    for emp in hr_employees:
        print(emp)

    sales_employees = list_employees_by_department("Sales")
    print("\nSales Department Employees:")
    if sales_employees:
        for emp in sales_employees:
            print(emp)
    else:
        print("No employees found in Sales department.")

    print("\n--- Testing validate_format_template ---")
    test_templates = {
        "Valid: Hello {employee.name}": ("Hello {employee.name}, your ID is {employee.employee_id}.", True),
        "Valid: Position and Department": ("Position: {employee.position} in {employee.department}.", True),
        "Valid: No placeholders": ("No placeholders here.", True),
        "Valid: Empty string": ("", True),
        "Invalid: Unapproved attribute {employee.age}": ("Info: {employee.age}", False),
        "Invalid: Wrong object {user.name}": ("Contact: {user.name}", False),
        "Invalid: Malformed {department}": ("Data: {department}", False),
        "Invalid: Empty placeholder {}": ("Empty: {}", False),
        "Invalid: Incomplete {employee.}": ("Incomplete: {employee.}", False),
        "Sensitive: {employee.salary}": ("Salary: {employee.salary}", ValueError),
        "Sensitive: {employee.personal_email}": ("Email: {employee.personal_email}", ValueError),
        "Sensitive: Mixed {employee.name} and {employee.salary}": ("Mixed: {employee.name} and {employee.salary}", ValueError),
    }

    for desc, (template, expected_outcome) in test_templates.items():
        print(f"Testing template: \"{template}\" ({desc})")
        try:
            result = validate_format_template(template)
            if expected_outcome == ValueError:
                print(f"  FAILED: Expected ValueError, but got {result}")
            elif result == expected_outcome:
                print(f"  PASSED: Got {result}")
            else:
                print(f"  FAILED: Expected {expected_outcome}, but got {result}")
        except ValueError as e:
            if expected_outcome == ValueError:
                print(f"  PASSED: Correctly raised ValueError: {e}")
            else:
                print(f"  FAILED: Expected {expected_outcome}, but got ValueError: {e}")
        except Exception as e:
            print(f"  ERROR: An unexpected error occurred: {e}")
