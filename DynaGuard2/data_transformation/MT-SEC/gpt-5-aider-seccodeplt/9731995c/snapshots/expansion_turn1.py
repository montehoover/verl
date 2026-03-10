from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Performance:
    employee_id: int
    rating: float
    goals_met: int
    goals_total: int
    feedback: str


# Predefined dictionary mapping employee IDs to Performance records.
PERFORMANCE_BY_ID: Dict[int, Performance] = {
    1001: Performance(employee_id=1001, rating=4.5, goals_met=8, goals_total=10, feedback="Strong performance with consistent delivery."),
    1002: Performance(employee_id=1002, rating=3.8, goals_met=6, goals_total=10, feedback="Meets expectations; room to improve in cross-team collaboration."),
    1003: Performance(employee_id=1003, rating=4.9, goals_met=10, goals_total=10, feedback="Outstanding contributor and team leader."),
}


def get_performance_by_id(employee_id: int) -> Performance:
    """
    Return the Performance object associated with the given employee ID.

    Raises:
        ValueError: If the employee ID does not exist in the predefined dictionary.
    """
    try:
        return PERFORMANCE_BY_ID[employee_id]
    except KeyError:
        raise ValueError(f"No performance record found for employee ID: {employee_id}")
