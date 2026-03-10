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
    Employee("Diana Miller", "HR Manager", "HR", 90000, "diana@personal.com")
]

# Helper class to provide a safe view of employee data for formatting
class SafeEmployeeView:
    def __init__(self, employee: Employee):
        self.name = employee.name
        self.position = employee.position
        # Add other safe attributes like 'department' if needed by templates
        # self.department = employee.department

def build_team_directory(dept_name: str, format_template: str) -> str:
    """
    Builds a directory string for employees in a given department using a format template.

    Args:
        dept_name: The name of the department.
        format_template: A string template with placeholders like '{employee.name}'
                         or '{employee.position}'.

    Returns:
        A formatted string representing the directory for the department.

    Raises:
        ValueError: If the department has no employees, if the department does not exist,
                    or if the format_template attempts to access restricted attributes.
    """
    dept_employees = [e for e in employees if e.department == dept_name]

    if not dept_employees:
        # This covers both "department does not exist" and "department is empty"
        raise ValueError(f"No employees found in department '{dept_name}'.")

    output_parts = []
    for emp in dept_employees:
        safe_view = SafeEmployeeView(emp)
        try:
            # The template uses {employee.name}, {employee.position}
            # We pass safe_view as 'employee' to format()
            formatted_part = format_template.format(employee=safe_view)
            output_parts.append(formatted_part)
        except AttributeError:
            # This catches attempts to access attributes not on SafeEmployeeView (e.g., employee.salary)
            raise ValueError("Template attempts to access a restricted or non-existent employee attribute.")
        except Exception as e:
            # Catches other formatting errors (e.g., syntax errors in the template string itself)
            raise ValueError(f"Error formatting template: {e}")

    return "\n".join(output_parts)

if __name__ == '__main__':
    print("--- IT Department (Successful) ---")
    it_template = "Employee: {employee.name} - Role: {employee.position}"
    try:
        print(build_team_directory("IT", it_template))
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- HR Department (Successful, different template) ---")
    hr_template = "{employee.name}\n  Position: {employee.position}"
    try:
        print(build_team_directory("HR", hr_template))
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Sales Department (No Employees/Department Not Found) ---")
    try:
        # "Sales" department has no employees in the 'employees' list
        print(build_team_directory("Sales", it_template))
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- IT Department (Template with restricted attribute: salary) ---")
    restricted_template_salary = "Name: {employee.name}, Salary: {employee.salary}"
    try:
        print(build_team_directory("IT", restricted_template_salary))
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- IT Department (Template with restricted attribute: personal_email) ---")
    restricted_template_email = "Name: {employee.name}, Email: {employee.personal_email}"
    try:
        print(build_team_directory("IT", restricted_template_email))
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- IT Department (Template with non-existent attribute on SafeView) ---")
    non_existent_attr_template = "Name: {employee.name}, Hobby: {employee.hobby}"
    try:
        print(build_team_directory("IT", non_existent_attr_template))
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- IT Department (Malformed Template Syntax) ---")
    malformed_template = "Name: {employee.name" # Missing closing brace
    try:
        print(build_team_directory("IT", malformed_template))
    except ValueError as e:
        print(f"Error: {e}")
