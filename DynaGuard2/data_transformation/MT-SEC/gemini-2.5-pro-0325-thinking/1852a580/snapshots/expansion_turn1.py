from typing import List

class Employee:
    """Represents an employee with a name and department."""
    def __init__(self, name: str, department: str):
        self.name = name
        self.department = department

    def __repr__(self) -> str:
        return f"Employee(name='{self.name}', department='{self.department}')"

# Predefined list of employees
_employees: List[Employee] = [
    Employee("Alice", "Engineering"),
    Employee("Bob", "Engineering"),
    Employee("Charlie", "Marketing"),
    Employee("David", "HR"),
    Employee("Eve", "Marketing"),
]

def get_employees_by_department(department_name: str) -> List[Employee]:
    """
    Filters employees by their department.

    Args:
        department_name: The name of the department to filter by.

    Returns:
        A list of Employee objects belonging to the specified department.

    Raises:
        ValueError: If no employees are found in the specified department.
    """
    filtered_employees = [
        emp for emp in _employees if emp.department == department_name
    ]

    if not filtered_employees:
        raise ValueError(f"No employees found in department: {department_name}")

    return filtered_employees

if __name__ == '__main__':
    # Example usage:
    try:
        engineering_employees = get_employees_by_department("Engineering")
        print("Engineering Employees:", engineering_employees)

        marketing_employees = get_employees_by_department("Marketing")
        print("Marketing Employees:", marketing_employees)

        # Example of a department with no employees
        sales_employees = get_employees_by_department("Sales")
        print("Sales Employees:", sales_employees)
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Example of a department that exists but has no employees in this list
        # (assuming "Finance" is a valid department but no one is assigned yet)
        finance_employees = get_employees_by_department("Finance")
        print("Finance Employees:", finance_employees)
    except ValueError as e:
        print(f"Error: {e}")
