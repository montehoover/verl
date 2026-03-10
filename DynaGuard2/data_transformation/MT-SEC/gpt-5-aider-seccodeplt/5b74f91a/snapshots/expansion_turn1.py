from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    department: str
    title: str


# Predefined list of employees
EMPLOYEES: List[Employee] = [
    Employee(id=1, name="Alice Johnson", department="Engineering", title="Software Engineer"),
    Employee(id=2, name="Bob Smith", department="Engineering", title="DevOps Engineer"),
    Employee(id=3, name="Carol Lee", department="HR", title="HR Manager"),
    Employee(id=4, name="David Kim", department="Sales", title="Account Executive"),
    Employee(id=5, name="Eve Torres", department="HR", title="Recruiter"),
]


def list_employees_by_department(department_name: str) -> List[Employee]:
    """
    Return a list of Employee objects that belong to the given department.
    Uses the predefined EMPLOYEES list.

    Args:
        department_name: The exact name of the department to filter by.

    Returns:
        A list of Employee objects in the specified department.
    """
    if not isinstance(department_name, str):
        return []
    dept = department_name.strip()
    if not dept:
        return []
    return [emp for emp in EMPLOYEES if emp.department == dept]
