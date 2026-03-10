from dataclasses import dataclass
from typing import Dict, Union


@dataclass(frozen=True)
class Employee:
    id: int
    name: str
    department: str
    title: str
    email: str


# Predefined dictionary of employees
EMPLOYEES: Dict[int, Employee] = {
    1001: Employee(id=1001, name="Alice Johnson", department="Engineering", title="Software Engineer", email="alice.johnson@example.com"),
    1002: Employee(id=1002, name="Bob Smith", department="Marketing", title="Marketing Manager", email="bob.smith@example.com"),
    1003: Employee(id=1003, name="Carol Davis", department="Human Resources", title="HR Specialist", email="carol.davis@example.com"),
}


def get_employee_by_id(employee_id: Union[int, str]) -> Employee:
    """
    Retrieve an Employee by ID.

    Args:
        employee_id: The ID of the employee (int or numeric string).

    Returns:
        The corresponding Employee object.

    Raises:
        ValueError: If the ID is invalid or does not exist.
    """
    try:
        emp_id = int(employee_id)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid employee ID: {employee_id!r}")

    try:
        return EMPLOYEES[emp_id]
    except KeyError:
        raise ValueError(f"Employee ID {emp_id} does not exist")
