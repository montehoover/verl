import string
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary  # Restricted
        self.personal_email = personal_email  # Restricted

employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com"),
    Employee("David Brown", "HR Specialist", "HR", 70000, "david@personal.com")
]

# --- Helper Class and Functions ---

RESTRICTED_ATTRIBUTES = {"salary", "personal_email"}
ALLOWED_ATTRIBUTES = {"name", "position", "department"}

class _EmployeeProxy:
    """A proxy object to safely expose only allowed employee attributes."""
    def __init__(self, emp_obj):
        for attr_name in ALLOWED_ATTRIBUTES:
            if hasattr(emp_obj, attr_name):
                setattr(self, attr_name, getattr(emp_obj, attr_name))

def _validate_template_access(list_template: str):
    """
    Pre-checks the template for restricted field access.
    Raises ValueError if restricted fields are accessed.
    """
    formatter = string.Formatter()
    try:
        for _, field_name, _, _ in formatter.parse(list_template):
            if not field_name:
                continue

            parts = field_name.split('.')
            if not (len(parts) == 2 and parts[0] == "employee"):
                # This field is not of the form 'employee.attribute', so we don't validate it here.
                # string.format() will handle it later (e.g., raise KeyError if placeholder not provided).
                continue

            attribute = parts[1]
            if attribute in RESTRICTED_ATTRIBUTES:
                log_message = (
                    f"Attempt to access restricted field '{attribute}' in template. "
                    f"Template: \"{list_template}\""
                )
                logging.warning(log_message)
                raise ValueError(
                    f"Access to restricted field '{attribute}' in template is not allowed."
                )
            # Optional: Check if attribute is unknown (neither allowed nor restricted)
            # if attribute not in ALLOWED_ATTRIBUTES:
            #     logging.warning(f"Template attempts to access unknown field 'employee.{attribute}'.")
            #     # Depending on strictness, could raise ValueError here too.
            #     # For now, we allow unknown attributes to be handled by the formatting step (AttributeError).

    except ValueError: # Re-raise ValueError specifically to not mask it as "Invalid template format"
        raise
    except Exception as e: # Catch potential errors from formatter.parse itself
        raise ValueError(f"Invalid template format or error during parsing: {e}") from e

def _filter_employees_by_department(employees_list: list[Employee], team_name: str) -> list[Employee]:
    """Filters a list of employees by their department."""
    return [emp for emp in employees_list if emp.department == team_name]

def _format_employee_entry(employee_obj: Employee, list_template: str) -> str:
    """
    Formats a single employee's details using the template.
    Uses _EmployeeProxy to restrict attribute access.
    Raises ValueError for formatting errors.
    """
    proxy = _EmployeeProxy(employee_obj)
    try:
        return list_template.format(employee=proxy)
    except AttributeError as e:
        raise ValueError(f"Error accessing attribute in template for employee '{employee_obj.name}': {e}") from e
    except KeyError as e:
        raise ValueError(f"Error formatting template due to missing key: {e}") from e
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during template formatting for employee '{employee_obj.name}': {e}") from e

# --- Main Function ---

def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Compiles an employee directory for a specified department based on a template.

    Args:
        team_name: The department name to filter by.
        list_template: The string template to populate with employee details.
                       Placeholders like '{employee.name}' are supported.

    Returns:
        A formatted string representing the employee directory for the specified department.

    Raises:
        ValueError: If the template attempts to access restricted fields (salary, personal_email),
                    or if an unexpected error occurs during template processing.
    """
    _validate_template_access(list_template)

    department_employees = _filter_employees_by_department(employees, team_name)

    formatted_entries = []
    for emp in department_employees:
        entry = _format_employee_entry(emp, list_template)
        formatted_entries.append(entry)

    return "\n".join(formatted_entries)

if __name__ == '__main__':
    # Example Usage:
    it_template = "Name: {employee.name}, Position: {employee.position}, Department: {employee.department}"
    hr_template = "- {employee.name} ({employee.position})"

    print("IT Department:")
    try:
        it_directory = build_worker_list("IT", it_template)
        print(it_directory)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nHR Department:")
    try:
        hr_directory = build_worker_list("HR", hr_template)
        print(hr_directory)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nAttempting to access restricted field (salary):")
    restricted_template = "Name: {employee.name}, Salary: {employee.salary}"
    try:
        restricted_directory = build_worker_list("IT", restricted_template)
        print(restricted_directory)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nAttempting to access non-existent field:")
    non_existent_field_template = "Name: {employee.name}, Nickname: {employee.nickname}"
    try:
        non_existent_directory = build_worker_list("IT", non_existent_field_template)
        print(non_existent_directory)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nAttempting to use a malformed template (KeyError):")
    malformed_template = "Details: {details}" # 'details' is not 'employee.something'
    try:
        malformed_directory = build_worker_list("IT", malformed_template)
        print(malformed_directory)
    except ValueError as e:
        print(f"Error: {e}")
