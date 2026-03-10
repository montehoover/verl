from dataclasses import dataclass
from datetime import date
from typing import Dict, Union


@dataclass(frozen=True)
class Performance:
    employee_id: str
    rating: int
    goals_met: int
    last_review_date: date
    notes: str = ""


# Predefined dictionary of employee performances
PERFORMANCES: Dict[str, Performance] = {
    "E001": Performance(employee_id="E001", rating=5, goals_met=12, last_review_date=date(2025, 6, 1), notes="Outstanding performer"),
    "E002": Performance(employee_id="E002", rating=3, goals_met=7, last_review_date=date(2025, 5, 20), notes="Meets expectations"),
    "E003": Performance(employee_id="E003", rating=4, goals_met=9, last_review_date=date(2025, 4, 18), notes="Strong contributor"),
}


def get_performance_data(employee_id: Union[str, int]) -> Performance:
    """
    Fetch the Performance object for the given employee ID.

    Args:
        employee_id: The employee identifier (string or integer).

    Returns:
        Performance: The performance record for the given employee.

    Raises:
        ValueError: If the employee ID does not exist.
    """
    key = str(employee_id).strip()
    if key in PERFORMANCES:
        return PERFORMANCES[key]
    raise ValueError(f"Employee ID does not exist: {employee_id}")
