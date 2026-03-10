from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    department: str
    title: Optional[str] = None


# Predefined list of employees
EMPLOYEES: List[Employee] = [
    Employee(id=1, name="Alice Johnson", department="Engineering", title="Software Engineer"),
    Employee(id=2, name="Bob Smith", department="Engineering", title="DevOps Engineer"),
    Employee(id=3, name="Carol Perez", department="HR", title="HR Manager"),
    Employee(id=4, name="David Lee", department="Sales", title="Account Executive"),
    Employee(id=5, name="Eve Kim", department="Marketing", title="Marketing Specialist"),
    Employee(id=6, name="Frank Miller", department="Finance", title="Financial Analyst"),
]


def get_employees_by_department(department: str) -> List[Employee]:
    """
    Return a list of Employee objects that belong to the given department.

    Raises:
        ValueError: If the department is empty or does not exist.
    """
    if not isinstance(department, str) or not department.strip():
        raise ValueError("Department name must be a non-empty string.")

    dept_norm = department.strip().lower()
    existing_departments = {e.department.lower() for e in EMPLOYEES}

    if dept_norm not in existing_departments:
        raise ValueError(f"Department '{department}' does not exist.")

    return [e for e in EMPLOYEES if e.department.lower() == dept_norm]
