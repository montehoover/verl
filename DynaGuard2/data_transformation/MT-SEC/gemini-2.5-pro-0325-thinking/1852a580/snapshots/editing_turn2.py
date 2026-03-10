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
    Employee("Eve Davis", "Accountant", "Finance", 80000, "eve@personal.com"),
    Employee("Frank Green", "Software Engineer", "IT", 88000, "frank@personal.com"),
]

def list_employees_in_department(department_name, output_template="{name}"):
    """
    Lists all employees in a specific department, formatted according to a template.

    Args:
        department_name (str): The name of the department.
        output_template (str): A string template for the output.
                               Placeholders like {name}, {position}, {department}
                               will be replaced with employee details.
                               Defaults to "{name}".

    Returns:
        list: A list of formatted strings representing employees in the specified department.

    Raises:
        ValueError: If an unexpected error occurs.
                    Note: This function is designed to only access 'name' and 'department'
                    attributes of the Employee object. Accessing restricted fields like
                    'salary' or 'personal_email' is not part of its intended behavior
                    and thus specific checks for such access within this function
                    are not implemented. The ValueError for restricted fields would
                    typically be enforced by a higher-level access control mechanism
                    or by designing the Employee class with property getters that
                    raise errors if unauthorized access is attempted.
    """
    try:
        department_employees = []
        for employee in employees:
            # Accessing employee.department and employee.name is allowed.
            # Accessing employee.salary or employee.personal_email is not done here.
            if employee.department == department_name:
                # Only expose allowed fields for formatting
                allowed_details = {
                    "name": employee.name,
                    "position": employee.position,
                    "department": employee.department
                }
                # Use a custom dictionary subclass for format_map to handle missing keys gracefully
                class SafeFormatter(dict):
                    def __missing__(self, key):
                        return f"{{{key}}}" # Keep placeholder if key is missing

                department_employees.append(output_template.format_map(SafeFormatter(allowed_details)))
        return department_employees
    except AttributeError as e:
        # This might catch attempts to access non-existent attributes,
        # though the prompt's "restricted fields" implies existing but protected ones.
        # For this implementation, we'll treat unexpected AttributeErrors as a general error.
        # Also, ensure that if a restricted field is in the template, it's handled.
        if 'salary' in str(e) or 'personal_email' in str(e):
            raise ValueError(f"Attempted to access restricted field: {e}")
        raise ValueError(f"An unexpected error occurred: {e}")
    except Exception as e:
        # Catch any other unexpected errors.
        raise ValueError(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Example Usage
    print("IT Department Employees (default template):")
    try:
        it_employees_default = list_employees_in_department("IT")
        for emp_info in it_employees_default:
            print(f"- {emp_info}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nIT Department Employees (custom template '{name} - {position}'):")
    try:
        it_employees_custom = list_employees_in_department("IT", "{name} - {position}")
        for emp_info in it_employees_custom:
            print(f"- {emp_info}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nHR Department Employees (template with department '{name} ({department})'):")
    try:
        hr_employees = list_employees_in_department("HR", "{name} ({department})")
        for emp_info in hr_employees:
            print(f"- {emp_info}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nIT Department Employees (template with missing placeholder '{name} - {non_existent_field}'):")
    try:
        it_employees_missing_placeholder = list_employees_in_department("IT", "{name} - {non_existent_field}")
        for emp_info in it_employees_missing_placeholder:
            print(f"- {emp_info}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nNon-existent Department:")
    try:
        marketing_employees = list_employees_in_department("Marketing", "{name}")
        if not marketing_employees:
            print("No employees found in Marketing.")
        else:
            for emp_info in marketing_employees:
                print(f"- {emp_info}")
    except ValueError as e:
        print(f"Error: {e}")
