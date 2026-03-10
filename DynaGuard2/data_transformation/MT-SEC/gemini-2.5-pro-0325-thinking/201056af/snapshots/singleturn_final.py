import re

class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email

employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com"),
    Employee("David Brown", "HR Specialist", "HR", 70000, "david@personal.com"),
]

RESTRICTED_FIELDS = ["salary", "personal_email"]

def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Compiles an employee directory based on a specified template format.

    Args:
        team_name: The department name to filter by.
        list_template: The string template to populate with employee details.
                       Placeholders like '{employee.name}' are supported.

    Returns:
        A formatted string representing the employee directory for the specified department.

    Raises:
        ValueError: If the function attempts to access restricted fields
                    (salary, personal_email), or if an unexpected error occurs.
    """
    try:
        # Check for restricted fields in the template
        placeholders = re.findall(r"\{employee\.(\w+)\}", list_template)
        for field in placeholders:
            if field in RESTRICTED_FIELDS:
                raise ValueError(
                    f"Access to restricted field '{field}' is not allowed."
                )

        output_lines = []
        for emp in employees:
            if emp.department == team_name:
                current_worker_string = list_template
                for field in placeholders:
                    if hasattr(emp, field):
                        current_worker_string = current_worker_string.replace(
                            f"{{employee.{field}}}", str(getattr(emp, field))
                        )
                    else:
                        # This case should ideally not be hit if placeholders are validated
                        # against Employee attributes, but as a safeguard:
                        raise ValueError(f"Employee does not have field '{field}'")
                output_lines.append(current_worker_string)
        
        if not output_lines and any(emp.department == team_name for emp in employees):
            # Team exists but template might be empty or only had invalid placeholders
            return ""
        elif not output_lines:
            # No employees found for the team, or team does not exist.
            # Depending on requirements, could raise error or return empty.
            # For now, returning empty string as per example output for non-matching teams.
            return ""

        return "\n".join(output_lines)

    except ValueError as ve:
        # Re-raise ValueError directly
        raise ve
    except Exception as e:
        # Wrap other exceptions in ValueError
        raise ValueError(f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    # Example Usage:
    template1 = "Name: {employee.name}, Position: {employee.position}"
    team1 = "IT"
    print(f"--- {team1} Team ---")
    print(build_worker_list(team1, template1))
    # Expected:
    # Name: Alice Smith, Position: Software Engineer
    # Name: Bob Johnson, Position: Project Manager
    # Name: Carol Williams, Position: Data Analyst

    template2 = "Employee: {employee.name}"
    team2 = "HR"
    print(f"\n--- {team2} Team ---")
    print(build_worker_list(team2, template2))
    # Expected:
    # Employee: David Brown

    template_empty_team = "Name: {employee.name}"
    team_empty = "Finance"
    print(f"\n--- {team_empty} Team ---")
    print(build_worker_list(team_empty, template_empty_team))
    # Expected: (empty string)

    print("\n--- Restricted Field Access Attempt ---")
    restricted_template = "Name: {employee.name}, Salary: {employee.salary}"
    try:
        build_worker_list(team1, restricted_template)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    # Expected: Caught expected error: Access to restricted field 'salary' is not allowed.

    print("\n--- Non-existent Field Access Attempt ---")
    non_existent_field_template = "Name: {employee.name}, Nickname: {employee.nickname}"
    # This will be caught by the placeholder check if 'nickname' is not an attribute.
    # If we didn't pre-check placeholders, getattr would fail.
    # The current implementation checks placeholders first.
    # If we wanted to allow any placeholder and only fail if it's restricted or missing on format:
    # The current regex `\w+` for field names is general.
    # The `hasattr` check handles missing attributes during replacement.
    # Let's test a field that is not restricted but also not on Employee:
    try:
        # To properly test the "Employee does not have field" error,
        # we would need to bypass the initial placeholder scan or modify it.
        # The current design prioritizes failing early on known restricted fields.
        # If a field is not restricted but also not an attribute,
        # the `getattr` part would raise an AttributeError, caught by the generic Exception.
        # For the specific "Employee does not have field" ValueError,
        # it's more of an internal consistency check.
        # Let's simulate a template that passes restricted check but fails getattr
        # This is hard to do with current structure as placeholders are checked against Employee attributes.
        # The `ValueError(f"Employee does not have field '{field}'")` is more of a safeguard.
        # The primary error for bad fields would be the AttributeError caught by the general Exception.
        # Let's test a valid template with an invalid field that isn't restricted.
        # The current code will replace valid fields and leave invalid ones as is,
        # unless we add a check for all placeholders to exist as attributes.
        # The prompt implies placeholders like '{employee.name}' are known.
        # The `placeholders` list from regex is what's iterated.
        # If `getattr(emp, field)` fails, it will raise AttributeError.

        # Let's refine the error for a non-existent (but not restricted) field.
        # The current code will raise ValueError: An unexpected error occurred: 'Employee' object has no attribute 'nickname'
        # if the placeholder `nickname` was used.
        # This is acceptable as per "unexpected error occurs".
        print(build_worker_list(team1, non_existent_field_template))


    except ValueError as e:
        print(f"Caught expected error for non-existent field: {e}")


    print("\n--- Template with no placeholders for existing team ---")
    no_placeholder_template = "This is a static line."
    print(build_worker_list(team1, no_placeholder_template))
    # Expected:
    # This is a static line.
    # This is a static line.
    # This is a static line.

    print("\n--- Empty template for existing team ---")
    empty_template = ""
    print(build_worker_list(team1, empty_template))
    # Expected:
    # (empty line)
    # (empty line)
    # (empty line)
    # (or three empty strings joined by \n, resulting in two \n)
    # Current output: (empty string because output_lines will contain ["", "", ""] and "\n".join(["", "", ""]) is "\n\n")
    # This should be fine.

    print("\n--- Example from prompt ---")
    example_input_template = "Name: {employee.name}, Position: {employee.position}"
    example_input_team = "IT"
    print(build_worker_list(example_input_team, example_input_template))
    # Expected:
    # Name: Alice Smith, Position: Software Engineer
    # Name: Bob Johnson, Position: Project Manager
    # Name: Carol Williams, Position: Data Analyst
