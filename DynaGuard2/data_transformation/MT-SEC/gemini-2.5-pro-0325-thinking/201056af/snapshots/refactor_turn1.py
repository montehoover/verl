import string

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
    restricted_attributes = {"salary", "personal_email"}
    allowed_attributes = {"name", "position", "department"}

    # Pre-check the template for restricted field access
    formatter = string.Formatter()
    try:
        for _, field_name, _, _ in formatter.parse(list_template):
            if field_name:
                parts = field_name.split('.')
                if len(parts) == 2 and parts[0] == "employee":
                    attribute = parts[1]
                    if attribute in restricted_attributes:
                        raise ValueError(
                            f"Access to restricted field '{attribute}' in template is not allowed."
                        )
                    # Optional: Check if attribute is unknown (neither allowed nor restricted)
                    # if attribute not in allowed_attributes:
                    #     raise ValueError(f"Access to unknown field '{attribute}' in template.")
                # Silently ignore field_names not matching 'employee.attr' pattern,
                # str.format will handle them (likely raising KeyError if not provided).
    except Exception as e: # Catch potential errors from formatter.parse itself, though unlikely for valid templates
        raise ValueError(f"Invalid template format: {e}") from e


    class EmployeeProxy:
        """A proxy object to safely expose only allowed employee attributes."""
        def __init__(self, emp_obj):
            for attr_name in allowed_attributes:
                if hasattr(emp_obj, attr_name):
                    setattr(self, attr_name, getattr(emp_obj, attr_name))
                # If emp_obj doesn't have an "allowed" attribute (e.g. department is None),
                # the proxy won't have it. Template access would then cause AttributeError.

    formatted_entries = []
    for emp in employees:
        if emp.department == team_name:
            proxy = EmployeeProxy(emp)
            try:
                # The template uses {employee.name}, so it expects an object named 'employee'
                # in the format call, and this object should have attributes like 'name'.
                current_entry = list_template.format(employee=proxy)
                formatted_entries.append(current_entry)
            except AttributeError as e:
                # Catches if template refers to employee.non_existent_attr
                # or employee.attr_not_on_proxy (e.g. an allowed attr missing on original employee)
                # or an attribute that was restricted and thus not added to proxy (though pre-check should catch this)
                raise ValueError(f"Error accessing attribute in template for employee '{emp.name}': {e}") from e
            except KeyError as e:
                # Catches if template refers to a placeholder not resolvable with `employee=proxy`
                # e.g. "{some_other_var}"
                raise ValueError(f"Error formatting template due to missing key: {e}") from e
            except Exception as e:
                # Catches other formatting errors (e.g., bad format specifier)
                raise ValueError(f"An unexpected error occurred during template formatting for employee '{emp.name}': {e}") from e

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
