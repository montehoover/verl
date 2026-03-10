from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class Employee:
    id: str
    name: str
    title: str
    department: str
    email: str | None = None


# Predefined dictionary of employees keyed by string worker ID
EMPLOYEES: Dict[str, Employee] = {
    "1001": Employee(id="1001", name="Alice Johnson", title="Software Engineer", department="Engineering", email="alice.johnson@example.com"),
    "1002": Employee(id="1002", name="Bob Smith", title="Product Manager", department="Product", email="bob.smith@example.com"),
    "1003": Employee(id="1003", name="Carol Lee", title="Data Analyst", department="Data", email="carol.lee@example.com"),
}


def get_employee_by_id(worker_id: Any) -> Employee:
    """
    Retrieve an Employee by their worker ID.

    Args:
        worker_id: The identifier of the employee (int, str, or any type convertible to str).

    Returns:
        The corresponding Employee object.

    Raises:
        ValueError: If no employee exists with the given worker ID.
    """
    key = str(worker_id)
    try:
        return EMPLOYEES[key]
    except KeyError as exc:
        raise ValueError(f"Employee with ID {worker_id} does not exist.") from exc
