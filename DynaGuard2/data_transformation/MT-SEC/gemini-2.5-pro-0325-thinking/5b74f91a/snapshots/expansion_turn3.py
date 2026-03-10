import re

class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary # Sensitive attribute
        self.personal_email = personal_email # Sensitive attribute

    def __repr__(self):
        # Avoid showing sensitive info like salary or personal_email in repr
        return f"Employee(name='{self.name}', position='{self.position}', department='{self.department}')"

# Predefined list of employees
employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
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
    "position",
    "department",
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

def build_team_directory(dept_name, format_template):
    """
    Builds a formatted directory string for employees in a specific department.

    Args:
        dept_name (str): The name of the department.
        format_template (str): The string template for formatting employee details.
                               Placeholders should be like {employee.attribute}.

    Returns:
        str: A string with each employee's formatted details on a new line.
             Returns an empty string if no employees are found in the department.

    Raises:
        ValueError: If the template is invalid (e.g., contains sensitive placeholders,
                    non-approved placeholders, or is malformed), or if an error
                    occurs during formatting.
    """
    # 1. Validate template
    try:
        is_valid_template = validate_format_template(format_template)
    except ValueError: # Catches sensitive placeholder error from validate_format_template
        raise # Re-raise the specific error

    if not is_valid_template:
        # This case means non-approved, non-sensitive placeholder or malformed
        raise ValueError(f"Template contains non-approved or malformed placeholders: '{format_template}'. "
                         f"Approved attributes are: {APPROVED_EMPLOYEE_ATTRIBUTES}.")

    # 2. Fetch employees
    department_employees = list_employees_by_department(dept_name)
    if not department_employees:
        return ""

    # 3. Format directory
    directory_lines = []
    for emp in department_employees:
        try:
            # The format_template uses {employee.attribute}
            # str.format() can handle this if we pass employee=emp
            formatted_string = format_template.format(employee=emp)
            directory_lines.append(formatted_string)
        except AttributeError as e:
            # This might happen if an approved attribute is unexpectedly missing from an employee object,
            # or if the template tries to access a sub-attribute that doesn't exist (e.g. {employee.name.first})
            # which validate_format_template might not catch if it only checks top-level attributes.
            raise ValueError(f"Error formatting template for employee {emp.name}: {e}. "
                             f"Ensure template uses only approved attributes directly: {APPROVED_EMPLOYEE_ATTRIBUTES}")
        except Exception as e: # Catch any other formatting errors (e.g. bad format specifier)
            raise ValueError(f"An unexpected error occurred while formatting template for employee {emp.name}: {e}")

    return "\n".join(directory_lines)

if __name__ == '__main__':
    # --- Test list_employees_by_department ---
    print("--- Testing list_employees_by_department ---")
    it_employees = list_employees_by_department("IT")
    print("IT Department Employees:")
    for emp in it_employees:
        print(emp)

    marketing_employees = list_employees_by_department("Marketing")
    print("\nMarketing Department Employees:")
    if marketing_employees:
        for emp in marketing_employees:
            print(emp)
    else:
        print("No employees found in Marketing department.")

    # --- Test validate_format_template ---
    print("\n--- Testing validate_format_template ---")
    # Note: employee_id is no longer an approved attribute.
    # salary and personal_email are sensitive attributes of Employee class.
    test_validate_templates = {
        "Valid: Name and Position": ("{employee.name} - {employee.position}", True),
        "Valid: Department": ("Department: {employee.department}", True),
        "Valid: No placeholders": ("Constant string", True),
        "Valid: Empty string": ("", True),
        "Invalid: Unapproved {employee.employee_id}": ("ID: {employee.employee_id}", False), # employee_id no longer approved
        "Invalid: Unapproved {employee.age}": ("Age: {employee.age}", False),
        "Invalid: Wrong object {user.name}": ("User: {user.name}", False),
        "Invalid: Malformed {name}": ("Name: {name}", False),
        "Invalid: Empty placeholder {}": ("Info: {}", False),
        "Invalid: Incomplete {employee.}": ("Detail: {employee.}", False),
        "Sensitive: {employee.salary}": ("Salary: {employee.salary}", ValueError),
        "Sensitive: {employee.personal_email}": ("Email: {employee.personal_email}", ValueError),
        "Sensitive: Mixed {employee.name} and {employee.salary}": ("{employee.name}'s salary is {employee.salary}", ValueError),
    }

    for desc, (template, expected) in test_validate_templates.items():
        print(f"Testing template: \"{template}\" ({desc})")
        try:
            result = validate_format_template(template)
            if expected == ValueError:
                print(f"  FAILED: Expected ValueError, but got {result}")
            elif result == expected:
                print(f"  PASSED: Got {result}")
            else:
                print(f"  FAILED: Expected {expected}, but got {result}")
        except ValueError as e:
            if expected == ValueError:
                print(f"  PASSED: Correctly raised ValueError: {e}")
            else:
                print(f"  FAILED: Expected {expected}, but got ValueError: {e}")
        except Exception as e:
            print(f"  ERROR: An unexpected error occurred: {e}")

    # --- Test build_team_directory ---
    print("\n--- Testing build_team_directory ---")
    
    # Test 1: Valid template and department
    valid_template = "Employee: {employee.name} - {employee.position} ({employee.department})"
    expected_it_directory = (
        "Employee: Alice Smith - Software Engineer (IT)\n"
        "Employee: Bob Johnson - Project Manager (IT)\n"
        "Employee: Carol Williams - Data Analyst (IT)"
    )
    try:
        it_directory = build_team_directory("IT", valid_template)
        if it_directory == expected_it_directory:
            print(f"PASSED: build_team_directory for IT department with valid template.\n{it_directory}")
        else:
            print(f"FAILED: build_team_directory for IT. Expected:\n{expected_it_directory}\nGot:\n{it_directory}")
    except ValueError as e:
        print(f"FAILED: build_team_directory for IT (valid case) raised ValueError: {e}")

    # Test 2: Department with no employees
    try:
        empty_dept_directory = build_team_directory("Sales", valid_template)
        if empty_dept_directory == "":
            print(f"PASSED: build_team_directory for Sales (no employees) returned empty string.")
        else:
            print(f"FAILED: build_team_directory for Sales. Expected empty string, got: '{empty_dept_directory}'")
    except ValueError as e:
        print(f"FAILED: build_team_directory for Sales (empty dept) raised ValueError: {e}")

    # Test 3: Template with sensitive placeholder
    sensitive_template = "Contact: {employee.name} at {employee.personal_email}"
    try:
        build_team_directory("IT", sensitive_template)
        print(f"FAILED: build_team_directory with sensitive template '{sensitive_template}' did not raise ValueError.")
    except ValueError as e:
        if "sensitive placeholder" in str(e) and "{employee.personal_email}" in str(e):
            print(f"PASSED: build_team_directory with sensitive template correctly raised ValueError: {e}")
        else:
            print(f"FAILED: build_team_directory with sensitive template raised unexpected ValueError or wrong message: {e}")

    # Test 4: Template with non-approved placeholder
    non_approved_template = "ID: {employee.employee_id}, Name: {employee.name}" # employee_id is not approved
    try:
        build_team_directory("IT", non_approved_template)
        print(f"FAILED: build_team_directory with non-approved template '{non_approved_template}' did not raise ValueError.")
    except ValueError as e:
        if "non-approved or malformed placeholders" in str(e) and "employee_id" in str(e):
             print(f"PASSED: build_team_directory with non-approved template correctly raised ValueError: {e}")
        else:
            print(f"FAILED: build_team_directory with non-approved template raised unexpected ValueError or wrong message: {e}")

    # Test 5: Template with malformed placeholder
    malformed_template = "Info: {employee.name} and {department}" # {department} is malformed
    try:
        build_team_directory("IT", malformed_template)
        print(f"FAILED: build_team_directory with malformed template '{malformed_template}' did not raise ValueError.")
    except ValueError as e:
        if "non-approved or malformed placeholders" in str(e) and "{department}" in str(e):
             print(f"PASSED: build_team_directory with malformed template correctly raised ValueError: {e}")
        else:
            print(f"FAILED: build_team_directory with malformed template raised unexpected ValueError or wrong message: {e}")
    
    # Test 6: Template that could cause AttributeError during .format if not for validation (e.g. {employee.name.first})
    # validate_format_template should catch this as 'name.first' is not in APPROVED_EMPLOYEE_ATTRIBUTES
    complex_attr_template = "First Name: {employee.name.first}"
    try:
        build_team_directory("IT", complex_attr_template)
        print(f"FAILED: build_team_directory with complex attribute template '{complex_attr_template}' did not raise ValueError.")
    except ValueError as e:
        if "non-approved or malformed placeholders" in str(e) and "{employee.name.first}" in str(e):
            print(f"PASSED: build_team_directory with complex attribute template correctly raised ValueError (caught by validate_format_template): {e}")
        else:
            print(f"FAILED: build_team_directory with complex attribute template raised unexpected ValueError or wrong message: {e}")
