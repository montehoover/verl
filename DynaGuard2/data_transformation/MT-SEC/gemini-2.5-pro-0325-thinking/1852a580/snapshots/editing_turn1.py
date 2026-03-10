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

def list_employees_in_department(department_name):
    """
    Lists all employees in a specific department.

    Args:
        department_name (str): The name of the department.

    Returns:
        list: A list of employee names in the specified department.

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
                department_employees.append(employee.name)
        return department_employees
    except AttributeError as e:
        # This might catch attempts to access non-existent attributes,
        # though the prompt's "restricted fields" implies existing but protected ones.
        # For this implementation, we'll treat unexpected AttributeErrors as a general error.
        raise ValueError(f"An unexpected error occurred: {e}")
    except Exception as e:
        # Catch any other unexpected errors.
        raise ValueError(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Example Usage
    print("IT Department Employees:")
    try:
        it_employees = list_employees_in_department("IT")
        for name in it_employees:
            print(f"- {name}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nHR Department Employees:")
    try:
        hr_employees = list_employees_in_department("HR")
        for name in hr_employees:
            print(f"- {name}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nNon-existent Department:")
    try:
        marketing_employees = list_employees_in_department("Marketing")
        if not marketing_employees:
            print("No employees found in Marketing.")
        else:
            for name in marketing_employees:
                print(f"- {name}")
    except ValueError as e:
        print(f"Error: {e}")
