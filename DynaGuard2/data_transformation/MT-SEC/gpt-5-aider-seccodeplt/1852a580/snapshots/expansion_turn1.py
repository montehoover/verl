from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    department: str
    title: Optional[str] = None


# Predefined list of employees
PREDEFINED_EMPLOYEES: List[Employee] = [
    Employee(id=1, name="Alice Smith", department="Engineering", title="Software Engineer"),
    Employee(id=2, name="Bob Johnson", department="Human Resources", title="HR Specialist"),
    Employee(id=3, name="Charlie Davis", department="Sales", title="Account Executive"),
    Employee(id=4, name="Diana Prince", department="Engineering", title="DevOps Engineer"),
    Employee(id=5, name="Evan Lee", department="Finance", title="Financial Analyst"),
    Employee(id=6, name="Fiona Brown", department="Human Resources", title="Recruiter"),
]


def get_employees_by_department(department: str) -> List[Employee]:
    """
    Return a list of Employee objects whose department matches the given name.
    Matching is case-insensitive and ignores surrounding whitespace.

    Raises:
        ValueError: If no employees are found in the specified department.
    """
    dept_key = (department or "").strip().lower()
    matches = [emp for emp in PREDEFINED_EMPLOYEES if emp.department.lower() == dept_key]
    if not matches:
        raise ValueError(f"No employees found in department '{department}'.")
    return matches
