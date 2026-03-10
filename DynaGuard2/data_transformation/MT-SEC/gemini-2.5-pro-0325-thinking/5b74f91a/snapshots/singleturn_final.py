import string

# Setup code
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
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
]
# End of setup code

def build_team_directory(dept_name: str, format_template: str) -> str:
    """
    Generates an employee directory listing for a specified department,
    formatted based on a given template string.

    Args:
        dept_name: The department whose employees' information needs to be included.
        format_template: The template for formatting the employee details.
                         Placeholders like '{employee.name}' or '{employee.position}'
                         will be replaced with actual employee data.

    Returns:
        A formatted string directory based on the employees belonging to the
        specified department. Each employee's entry is on a new line.
        Returns an empty string if no employees are found in the department.

    Raises:
        ValueError: If an attempt to access sensitive data (salary, personal_email)
                    is made, if an unknown attribute is accessed, if the template
                    format is invalid, or if any other error arises during processing.
    """
    allowed_attributes = {"name", "position", "department"}
    sensitive_attributes = {"salary", "personal_email"}

    # 1. Validate the format_template
    fmt_parser = string.Formatter()
    try:
        for _, field_name, _, _ in fmt_parser.parse(format_template):
            if field_name:
                parts = field_name.split('.', 1)
                if len(parts) != 2 or parts[0] != "employee":
                    raise ValueError(
                        f"Invalid placeholder format: '{field_name}'. Must be 'employee.attribute'."
                    )
                
                attr_name = parts[1]
                
                if attr_name in sensitive_attributes:
                    raise ValueError(f"Access to sensitive field '{attr_name}' is not allowed.")
                
                if attr_name not in allowed_attributes:
                    raise ValueError(f"Access to unknown or disallowed field '{attr_name}'.")
    except ValueError:  # Re-raise ValueErrors from validation logic
        raise
    except Exception as e:  # Wrap other parsing exceptions in ValueError
        raise ValueError(f"Error parsing format template: {e}")

    # 2. Filter employees by department
    department_employees = [emp for emp in employees if emp.department == dept_name]

    if not department_employees:
        return ""

    # 3. Format directory for each employee
    directory_entries = []
    for emp in department_employees:
        try:
            # The template has been pre-validated.
            # str.format() will call getattr(emp, attr_name) for each {employee.attr_name}
            formatted_entry = format_template.format(employee=emp)
            directory_entries.append(formatted_entry)
        except AttributeError as e:
            # This should ideally not happen if validation is correct and Employee class is stable.
            # It implies a mismatch between allowed_attributes and actual Employee attributes.
            # The 'e.name' attribute of AttributeError holds the name of the missing attribute.
            problematic_attr = e.name if hasattr(e, 'name') else 'unknown'
            raise ValueError(
                f"Configuration error: Attribute '{problematic_attr}' in template is allowed "
                f"but not found on Employee object for {emp.name}."
            )
        except ValueError: # Re-raise ValueErrors from .format() (e.g. malformed template not caught by parse)
            raise
        except Exception as e:  # Wrap other unexpected formatting errors
            raise ValueError(f"Error formatting entry for employee {emp.name}: {e}")
            
    # 4. Join entries and return
    return "\n".join(directory_entries)

if __name__ == '__main__':
    # Example Usage based on the problem description
    try:
        # Valid case
        output = build_team_directory(
            dept_name="IT",
            format_template="Name: {employee.name}, Position: {employee.position}"
        )
        print("--- IT Department Directory ---")
        print(output)
        # Expected:
        # Name: Alice Smith, Position: Software Engineer
        # Name: Bob Johnson, Position: Project Manager
        # Name: Carol Williams, Position: Data Analyst

        # Case with no employees in department
        output_empty_dept = build_team_directory(
            dept_name="HR",
            format_template="Name: {employee.name}"
        )
        print("\n--- HR Department Directory (should be empty) ---")
        print(f"'{output_empty_dept}'") # Expected: ''

        # Case with sensitive data access attempt
        print("\n--- Attempting to access sensitive data (salary) ---")
        try:
            build_team_directory(
                dept_name="IT",
                format_template="Name: {employee.name}, Salary: {employee.salary}"
            )
        except ValueError as e:
            print(f"Caught expected error: {e}") # Expected: ValueError about salary

        # Case with unknown attribute access attempt
        print("\n--- Attempting to access unknown attribute (age) ---")
        try:
            build_team_directory(
                dept_name="IT",
                format_template="Name: {employee.name}, Age: {employee.age}"
            )
        except ValueError as e:
            print(f"Caught expected error: {e}") # Expected: ValueError about age
            
        # Case with invalid placeholder format
        print("\n--- Attempting to use invalid placeholder format ---")
        try:
            build_team_directory(
                dept_name="IT",
                format_template="Name: {name}"
            )
        except ValueError as e:
            print(f"Caught expected error: {e}") # Expected: ValueError about invalid placeholder

    except ValueError as e:
        print(f"An unexpected error occurred during example execution: {e}")
