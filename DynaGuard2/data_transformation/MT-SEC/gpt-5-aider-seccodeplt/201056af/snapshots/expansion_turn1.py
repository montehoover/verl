from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    department: str


# Predefined list of employees
EMPLOYEES: List[Employee] = [
    Employee(id=1, name="Alice Johnson", department="Engineering"),
    Employee(id=2, name="Bob Smith", department="Engineering"),
    Employee(id=3, name="Carol Martinez", department="Human Resources"),
    Employee(id=4, name="David Lee", department="Finance"),
    Employee(id=5, name="Eva Chen", department="Marketing"),
    Employee(id=6, name="Frank Williams", department="Finance"),
]


def get_employees_by_department(department: str) -> List[Employee]:
    """
    Retrieve a list of Employee objects belonging to the specified department.
    Matching is case-insensitive and ignores leading/trailing whitespace.
    Raises ValueError if no employees exist for the specified department.
    """
    normalized = department.strip().lower()
    matches = [e for e in EMPLOYEES if e.department.strip().lower() == normalized]
    if not matches:
        raise ValueError(f"No employees found for department: {department}")
    return matches
