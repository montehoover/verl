class Employee:
    def __init__(self, name, department):
        self.name = name
        self.department = department

    def __repr__(self):
        return f"Employee(name='{self.name}', department='{self.department}')"

_employees = [
    Employee("Alice", "Engineering"),
    Employee("Bob", "Engineering"),
    Employee("Charlie", "Marketing"),
    Employee("Diana", "HR"),
    Employee("Eve", "Marketing"),
]

def get_employees_by_department(department_name: str) -> list[Employee]:
    """
    Retrieves a list of Employee objects belonging to the specified department.

    Args:
        department_name: The name of the department.

    Returns:
        A list of Employee objects.

    Raises:
        ValueError: If no employees exist for the specified department.
    """
    department_employees = [
        emp for emp in _employees if emp.department == department_name
    ]
    if not department_employees:
        raise ValueError(f"No employees found for department: {department_name}")
    return department_employees

if __name__ == '__main__':
    # Example usage:
    try:
        engineering_employees = get_employees_by_department("Engineering")
        print("Engineering Employees:", engineering_employees)

        marketing_employees = get_employees_by_department("Marketing")
        print("Marketing Employees:", marketing_employees)

        # Example of a department with no employees (should raise ValueError)
        sales_employees = get_employees_by_department("Sales")
        print("Sales Employees:", sales_employees)
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example with a non-existent department
        hr_employees = get_employees_by_department("HR")
        print("HR Employees:", hr_employees)
    except ValueError as e:
        print(f"Error: {e}")
