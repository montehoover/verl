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

class SafeFormatter(dict):
    """
    A dictionary subclass that returns the placeholder itself if a key is missing.
    Useful for str.format_map to gracefully handle missing placeholders.
    Example: "{name} is {age}".format_map(SafeFormatter({'name': 'Alice'}))
             will result in "Alice is {age}"
    """
    def __missing__(self, key):
        return f"{{{key}}}"

def list_employees_by_department(department, format_template):
    """
    Lists all employees in a given department, formatted according to a template.

    Args:
        department (str): The department to filter by.
        format_template (str): A string template with placeholders like {name}
                               and {position}.

    Returns:
        list: A list of formatted strings, one for each employee in the
              specified department.

    Raises:
        ValueError: If the department or format_template argument is invalid,
                    or if no employees are found in the specified department,
                    or if an attempt is made to access sensitive information.
    """
    if not isinstance(department, str) or not department:
        raise ValueError("Department must be a non-empty string.")
    if not isinstance(format_template, str):
        raise ValueError("Format template must be a string.")

    formatted_list = []
    employees_in_department = [e for e in employees if e.department == department]

    if not employees_in_department:
        raise ValueError(f"No employees found in the {department} department or department does not exist.")

    for employee in employees_in_department:
        # Check for attempts to access sensitive information (example)
        if hasattr(employee, 'salary') and employee.salary > 1000000: # Arbitrary check
             raise ValueError("Attempt to access or process overly sensitive information for an employee.")

        employee_data = {
            'name': employee.name,
            'position': employee.position
            # Add other non-sensitive fields here if they should be available to the template
        }
        
        try:
            formatted_string = format_template.format_map(SafeFormatter(employee_data))
            formatted_list.append(formatted_string)
        except Exception as e:
            # This might catch unexpected errors during formatting, though SafeFormatter handles KeyError
            raise ValueError(f"Error formatting template for employee {employee.name}: {e}")


    return formatted_list

if __name__ == '__main__':
    try:
        print("Listing IT department with standard format:")
        template1 = "Employee: {name} - Position: {position}"
        it_employees = list_employees_by_department("IT", template1)
        for entry in it_employees:
            print(entry)

        print("\nListing HR department with a format including a missing placeholder:")
        template2 = "{name} ({position}), Department: {dept_code}" # {dept_code} is missing
        hr_employees = list_employees_by_department("HR", template2)
        for entry in hr_employees:
            print(entry)

        # Example of a department that doesn't exist
        # print("\nAttempting to list non-existent Finance department:")
        # finance_employees = list_employees_by_department("Finance", template1)
        # for entry in finance_employees:
        #     print(entry)

        # Example of invalid input for department
        # print("\nAttempting to list with empty department:")
        # list_employees_by_department("", template1)

        # Example of invalid input for template
        # print("\nAttempting to list with non-string template:")
        # list_employees_by_department("IT", 123)

    except ValueError as e:
        print(f"Error: {e}")

    # Example demonstrating the sensitive info check (will require an employee with high salary)
    # employees.append(Employee("Sensitive Sam", "CEO", "Executive", 2000000, "sam@personal.com"))
    # try:
    #     print("\nListing Executive department (may trigger sensitive info error):")
    #     executive_employees = list_employees_by_department("Executive", template1)
    #     for entry in executive_employees:
    #         print(entry)
    # except ValueError as e:
    #     print(f"Error during Executive listing: {e}")
