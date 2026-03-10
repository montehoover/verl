from dataclasses import dataclass
from typing import List

@dataclass
class Employee:
    """Represents an employee with a name and department."""
    name: str
    department: str

# Predefined list of employees
_employees: List[Employee] = [
    Employee(name="Alice Wonderland", department="Engineering"),
    Employee(name="Bob The Builder", department="Engineering"),
    Employee(name="Charlie Brown", department="Marketing"),
    Employee(name="Diana Prince", department="Sales"),
    Employee(name="Edward Scissorhands", department="HR"),
    Employee(name="Fiona Gallagher", department="Marketing"),
]

# Set of all available departments
_departments: set[str] = {emp.department for emp in _employees}

def get_employees_by_department(department_name: str) -> List[Employee]:
    """
    Filters employees by department.

    Args:
        department_name: The name of the department to filter by.

    Returns:
        A list of Employee objects belonging to the specified department.

    Raises:
        ValueError: If the department_name does not exist.
    """
    if department_name not in _departments:
        raise ValueError(f"Department '{department_name}' does not exist. Available departments: {', '.join(sorted(list(_departments)))}")

    return [employee for employee in _employees if employee.department == department_name]

if __name__ == '__main__':
    # Example usage:
    try:
        engineering_employees = get_employees_by_department("Engineering")
        print("Engineering Employees:")
        for emp in engineering_employees:
            print(f"- {emp.name}")

        marketing_employees = get_employees_by_department("Marketing")
        print("\nMarketing Employees:")
        for emp in marketing_employees:
            print(f"- {emp.name}")

        # Example of a non-existent department
        print("\nTrying to get employees from a non-existent department:")
        non_existent_employees = get_employees_by_department("Finance")
    except ValueError as e:
        print(e)

    # Example: Get all unique department names
    print(f"\nAll available departments: {', '.join(sorted(list(_departments)))}")
