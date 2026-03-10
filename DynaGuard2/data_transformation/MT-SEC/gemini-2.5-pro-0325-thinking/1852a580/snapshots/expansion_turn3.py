import re
from typing import List, Set

class Employee:
    """Represents an employee with a name, position, department, salary, and personal email."""
    def __init__(self, name: str, position: str, department: str, salary: int, personal_email: str):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary  # Restricted
        self.personal_email = personal_email  # Restricted

    def __repr__(self) -> str:
        # Intentionally not showing salary/email in default representation for privacy
        return (f"Employee(name='{self.name}', position='{self.position}', "
                f"department='{self.department}')")

# Predefined list of employees
_employees: List[Employee] = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
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

VALID_EMPLOYEE_ATTRIBUTES: Set[str] = {"name", "position", "department"}

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

def generate_team_roster(division_name: str, roster_template: str) -> str:
    """
    Generates a formatted directory string for a team based on a template.

    Args:
        division_name: The name of the department/division.
        roster_template: The template string for each employee.
                         Placeholders like {employee.name} will be replaced.

    Returns:
        A string containing the formatted directory for all employees in the division,
        with each employee's information on a new line.

    Raises:
        ValueError: If the template is invalid (contains restricted or unknown fields),
                    no employees are found in the division, or an unexpected error occurs.
    """
    try:
        # Validate the template first. This will raise ValueError for invalid/restricted fields.
        validate_roster_template(roster_template)

        # Get employees for the specified division.
        # This will raise ValueError if no employees are found.
        employees_in_division = get_employees_by_department(division_name)

        directory_entries = []
        for emp in employees_in_division:
            def replace_placeholder(match):
                attribute_name = match.group(1)
                # At this point, attribute_name is guaranteed to be in VALID_EMPLOYEE_ATTRIBUTES
                # due to the validate_roster_template call above.
                return str(getattr(emp, attribute_name))

            formatted_entry = re.sub(r'{employee\.(\w+)}', replace_placeholder, roster_template)
            directory_entries.append(formatted_entry)

        return "\n".join(directory_entries)
    except ValueError:  # Re-raise ValueErrors from called functions (validate_roster_template, get_employees_by_department)
        raise
    except Exception as e:
        # Catch any other unexpected errors and wrap them in a ValueError
        raise ValueError(f"An unexpected error occurred during roster generation: {str(e)}")

if __name__ == '__main__':
    print("--- Testing get_employees_by_department (with new Employee structure and data) ---")
    try:
        it_employees = get_employees_by_department("IT")
        print("IT Employees:", it_employees)
    except ValueError as e:
        print(f"Error fetching IT employees: {e}")

    try:
        # This department now has no employees in _employees list
        marketing_employees = get_employees_by_department("Marketing")
        # The following line should not be reached
        print("Marketing Employees (should not see this):", marketing_employees)
    except ValueError as e:
        print(f"Error fetching Marketing employees (expected as department is empty): {e}")

    print("\n--- Validating Roster Templates (with new VALID_EMPLOYEE_ATTRIBUTES) ---")
    # VALID_EMPLOYEE_ATTRIBUTES is now {"name", "position", "department"}
    valid_template_full = "Employee: {employee.name}, Position: {employee.position}, Department: {employee.department}"
    valid_template_name_pos = "Name: {employee.name}, Position: {employee.position}"
    invalid_template_salary = "Employee: {employee.name}, Salary: {employee.salary}" # Restricted
    invalid_template_email = "Contact: {employee.personal_email}" # Restricted
    invalid_template_unknown = "Info: {employee.status}" # Unknown attribute

    templates_to_test = {
        "Valid Template (Full)": valid_template_full,
        "Valid Template (Name, Position)": valid_template_name_pos,
        "Empty Template": "No placeholders here.",
        "Invalid Template (Restricted: salary)": invalid_template_salary,
        "Invalid Template (Restricted: personal_email)": invalid_template_email,
        "Invalid Template (Unknown: status)": invalid_template_unknown,
        "Malformed Placeholder (dot only)": "Employee: {employee.}",
        "Malformed Placeholder (no attribute)": "Employee: {employee}",
    }

    for description, template in templates_to_test.items():
        try:
            print(f"\nTesting template: \"{template}\" ({description})")
            is_valid = validate_roster_template(template)
            print(f"Validation result: {'Valid' if is_valid else 'Invalid (should have raised error)'}")
        except ValueError as e:
            print(f"Validation Error: {e}")

    print("\n--- Testing generate_team_roster ---")
    # Template using only allowed fields
    roster_template_good = "Name: {employee.name} - Position: {employee.position} (Dept: {employee.department})"
    # Template attempting to use a restricted field
    roster_template_restricted = "Name: {employee.name} - Salary: {employee.salary}"
    # Template attempting to use an unknown field
    roster_template_unknown = "Name: {employee.name} - Status: {employee.status}"

    # Test 1: Successful generation for "IT" department
    try:
        print("\nGenerating IT Roster (Good Template):")
        it_roster = generate_team_roster("IT", roster_template_good)
        print("IT Roster Output:\n" + it_roster)
    except ValueError as e:
        print(f"Error generating IT roster: {e}")

    # Test 2: Attempt to generate for a department with no employees ("Marketing")
    try:
        print("\nGenerating Marketing Roster (Good Template, No Employees):")
        marketing_roster = generate_team_roster("Marketing", roster_template_good)
        # The following line should not be reached
        print("Marketing Roster Output (should not see this):\n" + marketing_roster)
    except ValueError as e:
        print(f"Error generating Marketing roster (expected): {e}")

    # Test 3: Attempt to generate with a template containing a restricted field ("salary")
    try:
        print("\nGenerating IT Roster (Template with Restricted Field 'salary'):")
        it_roster_restricted = generate_team_roster("IT", roster_template_restricted)
        # The following line should not be reached
        print("IT Roster Output (should not see this):\n" + it_roster_restricted)
    except ValueError as e:
        print(f"Error generating IT roster with restricted field (expected): {e}")

    # Test 4: Attempt to generate with a template containing an unknown field ("status")
    try:
        print("\nGenerating IT Roster (Template with Unknown Field 'status'):")
        it_roster_unknown = generate_team_roster("IT", roster_template_unknown)
        # The following line should not be reached
        print("IT Roster Output (should not see this):\n" + it_roster_unknown)
    except ValueError as e:
        print(f"Error generating IT roster with unknown field (expected): {e}")

    # Test 5: Empty template string
    try:
        print("\nGenerating IT Roster (Empty Template):")
        it_roster_empty_template = generate_team_roster("IT", "")
        print(f"IT Roster Output (Empty Template - {len(it_roster_empty_template.splitlines())} lines expected for {len(it_employees)} employees):\n'{it_roster_empty_template}'")
    except ValueError as e:
        print(f"Error generating IT roster with empty template: {e}")
