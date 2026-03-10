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

def list_employees_by_department(department):
    """
    Lists all employees in a given department.

    Args:
        department (str): The department to filter by.

    Prints:
        The name and position of each employee in the specified department.

    Raises:
        ValueError: If the department argument is invalid, or if no employees
                    are found in the specified department, or if an attempt is made
                    to access sensitive information (though this function only
                    prints name and position).
    """
    if not isinstance(department, str) or not department:
        raise ValueError("Department must be a non-empty string.")

    found_employees = False
    print(f"\nEmployees in {department} department:")
    for employee in employees:
        if employee.department == department:
            # Check for attempts to access sensitive information (example)
            # This is a bit contrived here as we are only printing name/position
            # but demonstrates the principle if other attributes were involved.
            if hasattr(employee, 'salary') and employee.salary > 1000000: # Arbitrary check
                 raise ValueError("Attempt to access or process overly sensitive information.")

            print(f"  Name: {employee.name}, Position: {employee.position}")
            found_employees = True

    if not found_employees:
        raise ValueError(f"No employees found in the {department} department or department does not exist.")

if __name__ == '__main__':
    try:
        list_employees_by_department("IT")
        list_employees_by_department("HR")
        # Example of a department that doesn't exist
        # list_employees_by_department("Finance")
        # Example of invalid input
        # list_employees_by_department("")
    except ValueError as e:
        print(f"Error: {e}")

    # Example demonstrating the sensitive info check (will require an employee with high salary)
    # employees.append(Employee("Sensitive Sam", "CEO", "Executive", 2000000, "sam@personal.com"))
    # try:
    #     list_employees_by_department("Executive")
    # except ValueError as e:
    #     print(f"Error during Executive listing: {e}")
