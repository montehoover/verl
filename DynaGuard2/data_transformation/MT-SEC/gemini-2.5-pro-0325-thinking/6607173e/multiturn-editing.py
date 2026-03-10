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
    Employee("Eve Davis", "Marketing Manager", "Marketing", 90000, "eve@personal.com"),
]

def create_employee_directory(department, template):
    """
    Creates a directory string for employees in a given department,
    formatted according to a template.

    Args:
        department (str): The department to filter by.
        template (str): A string template with placeholders like {employee.name}
                        and {employee.position}.

    Returns:
        str: A single string representing the directory, with each employee's
             details formatted and separated by newlines.

    Raises:
        ValueError: If the department or template argument is invalid,
                    or if no employees are found in the specified department,
                    or if an attempt is made to access sensitive information.
        AttributeError: If the template refers to an attribute not present on
                        the Employee object (e.g., {employee.non_existent_field}).
    """
    if not isinstance(department, str) or not department:
        raise ValueError("Department must be a non-empty string.")
    if not isinstance(template, str): # Assuming template cannot be empty either
        raise ValueError("Template must be a non-empty string.")

    employees_in_department = [e for e in employees if e.department == department]

    if not employees_in_department:
        raise ValueError(f"No employees found in the {department} department or department does not exist.")

    directory_entries = []
    for employee in employees_in_department:
        # Check for attempts to access sensitive information (example)
        if hasattr(employee, 'salary') and employee.salary > 1000000: # Arbitrary check
             raise ValueError(f"Attempt to access or process overly sensitive information for employee: {employee.name}.")

        try:
            # Pass the whole employee object to format, allowing {employee.attribute} access
            formatted_entry = template.format(employee=employee)
            directory_entries.append(formatted_entry)
        except AttributeError as e:
            raise AttributeError(f"Error formatting template for employee {employee.name}: {e}. Ensure template uses valid employee attributes like '{{employee.name}}'.")
        except Exception as e:
            # Catch other potential formatting errors, though AttributeError is most likely
            raise ValueError(f"Unexpected error formatting template for employee {employee.name}: {e}")

    return "\n".join(directory_entries)

if __name__ == '__main__':
    try:
        print("Creating IT department directory:")
        dir_template1 = "Employee Name: {employee.name}\nPosition: {employee.position}\n---"
        it_directory = create_employee_directory("IT", dir_template1)
        print(it_directory)

        print("\nCreating HR department directory:")
        dir_template2 = "HR Contact: {employee.name} ({employee.position})"
        hr_directory = create_employee_directory("HR", dir_template2)
        print(hr_directory)

        # Example: Department that doesn't exist
        # print("\nAttempting to create directory for non-existent Finance department:")
        # finance_directory = create_employee_directory("Finance", dir_template1)
        # print(finance_directory)

        # Example: Invalid department input
        # print("\nAttempting with empty department:")
        # create_employee_directory("", dir_template1)

        # Example: Invalid template input
        # print("\nAttempting with non-string template:")
        # create_employee_directory("IT", None)

        # Example: Template with a non-existent attribute (will raise AttributeError)
        # print("\nAttempting with template having non-existent attribute:")
        # bad_template = "{employee.name} - {employee.hobby}"
        # create_employee_directory("IT", bad_template)

    except (ValueError, AttributeError) as e:
        print(f"Error: {e}")

    # Example demonstrating the sensitive info check
    # employees.append(Employee("Sensitive Sam", "CEO", "Executive", 2000000, "sam@personal.com"))
    # try:
    #     print("\nCreating Executive directory (may trigger sensitive info error):")
    #     executive_dir = create_employee_directory("Executive", dir_template1)
    #     print(executive_dir)
    # except (ValueError, AttributeError) as e:
    #     print(f"Error during Executive directory creation: {e}")
