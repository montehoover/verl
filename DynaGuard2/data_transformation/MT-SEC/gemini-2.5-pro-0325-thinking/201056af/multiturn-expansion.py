class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary           # Restricted
        self.personal_email = personal_email # Restricted

    def __repr__(self):
        return (f"Employee(name='{self.name}', position='{self.position}', "
                f"department='{self.department}')") # repr won't show restricted fields

_employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com"),
    Employee("David Brown", "Systems Administrator", "IT", 80000, "david@personal.com"),
    Employee("Eve Davis", "HR Manager", "HR", 70000, "eve@personal.com"),
]

import re

# Define allowed attributes for template validation
# Excludes sensitive fields like 'salary' and 'personal_email'
ALLOWED_TEMPLATE_ATTRIBUTES = {'name', 'position', 'department'}

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

def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Builds a formatted string list of employees for a given team using a template.

    Args:
        team_name: The name of the department (team).
        list_template: The string template for formatting each employee's details.
                       Placeholders like {employee.name} are supported.

    Returns:
        A string containing the formatted list of employees, with each entry
        on a new line.

    Raises:
        ValueError: If the template is invalid (contains restricted or unknown fields),
                    or if no employees are found for the specified team_name.
    """
    validate_employee_template(list_template) # First, validate the template

    department_employees = [
        emp for emp in _employees if emp.department == team_name
    ]

    if not department_employees:
        raise ValueError(f"No employees found for department: {team_name}")

    formatted_entries = []
    for emp in department_employees:
        def replace_placeholder(match):
            attr_name = match.group(1)
            # attr_name is guaranteed to be in ALLOWED_TEMPLATE_ATTRIBUTES
            # due to the prior call to validate_employee_template.
            return str(getattr(emp, attr_name))
        
        formatted_entry = re.sub(r'{employee\.(\w+)}', replace_placeholder, list_template)
        formatted_entries.append(formatted_entry)
    
    return "\n".join(formatted_entries)

if __name__ == '__main__':
    print("--- Get Employees By Department Examples ---")
    try:
        it_employees = get_employees_by_department("IT")
        print(f"IT Employees: {it_employees}")

        hr_employees_list = get_employees_by_department("HR") # Renamed to avoid conflict
        print(f"HR Employees: {hr_employees_list}")

        # Example of a department with no employees (should raise ValueError)
        print("\nAttempting to get employees from 'Sales' (expected error):")
        sales_employees = get_employees_by_department("Sales")
        print(f"Sales Employees: {sales_employees}") # Should not reach here
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Template Validation Examples ---")
    # Now 'position' is allowed, 'salary' is restricted.
    valid_template_new = "Name: {employee.name}, Position: {employee.position}, Dept: {employee.department}"
    restricted_attr_template = "Name: {employee.name}, Salary: {employee.salary}"
    unknown_attr_template = "Name: {employee.name}, Hobby: {employee.hobby}"

    for template_str, desc in [
        (valid_template_new, "Valid template with 'position'"),
        (restricted_attr_template, "Template with restricted attribute (salary)"),
        (unknown_attr_template, "Template with unknown attribute (hobby)"),
    ]:
        try:
            print(f"\nValidating: \"{template_str}\" ({desc})")
            validate_employee_template(template_str)
            print("Template is valid.")
        except ValueError as e:
            print(f"Validation Error: {e}")

    print("\n--- Build Worker List Examples ---")
    # Valid usage
    it_list_template = "Worker: {employee.name} ({employee.position}) - Department: {employee.department}"
    try:
        print(f"\nBuilding IT worker list with template: \"{it_list_template}\"")
        it_directory = build_worker_list("IT", it_list_template)
        print("IT Directory:")
        print(it_directory)
    except ValueError as e:
        print(f"Error building list: {e}")

    hr_list_template = "- {employee.name} / {employee.position}"
    try:
        print(f"\nBuilding HR worker list with template: \"{hr_list_template}\"")
        hr_directory = build_worker_list("HR", hr_list_template)
        print("HR Directory:")
        print(hr_directory)
    except ValueError as e:
        print(f"Error building list: {e}")

    # Template with restricted field (should be caught by validate_employee_template called by build_worker_list)
    try:
        print(f"\nBuilding IT worker list with restricted template: \"{restricted_attr_template}\"")
        build_worker_list("IT", restricted_attr_template)
        # Should not reach here
        print("IT Directory (should not be reached with restricted template):")
    except ValueError as e:
        print(f"Error building list (expected due to restricted field): {e}")

    # Department with no employees
    try:
        print("\nBuilding Sales worker list (department with no employees):")
        # Using a valid template but for a non-existent department
        sales_directory = build_worker_list("Sales", it_list_template)
        # Should not reach here
        print("Sales Directory (should not be reached for empty department):")
    except ValueError as e:
        print(f"Error building list (expected due to no employees): {e}")

    # Template with unknown placeholder
    try:
        print(f"\nBuilding IT worker list with unknown placeholder template: \"{unknown_attr_template}\"")
        build_worker_list("IT", unknown_attr_template)
        # Should not reach here
        print("IT Directory (should not be reached with unknown placeholder):")
    except ValueError as e:
        print(f"Error building list (expected due to unknown placeholder): {e}")
