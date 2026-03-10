from dataclasses import dataclass
from datetime import date
from typing import Dict, Union
from string import Formatter


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


def validate_summary_template(template: str) -> bool:
    """
    Validate that all placeholders in the template are allowed and well-formed.

    Allowed roots:
      - employee.<attr> (any identifier except restricted ones)
      - performance.<attr> (must be a field on Performance and not restricted)

    Restricted attributes (cannot be accessed under any root):
      - feedback
      - bonus

    Returns True if valid, otherwise raises ValueError.
    """
    if not isinstance(template, str):
        raise TypeError("template must be a string")

    formatter = Formatter()
    allowed_roots = {"employee", "performance"}
    restricted_attrs = {"feedback", "bonus"}
    performance_allowed_attrs = set(Performance.__dataclass_fields__.keys())

    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue  # literal text or escaped braces

        name = field_name.strip()
        if not name:
            raise ValueError("Empty placeholder '{}' is not allowed")

        if "[" in name or "]" in name:
            raise ValueError(f"Invalid placeholder access using indexing in '{{{field_name}}}'")

        parts = name.split(".")
        if parts[0] not in allowed_roots:
            raise ValueError(f"Unknown placeholder root '{parts[0]}' in '{{{field_name}}}'")

        if len(parts) != 2:
            raise ValueError(f"Invalid placeholder '{{{field_name}}}'. Use '{{employee.name}}' or '{{performance.rating}}'")

        root, attr = parts
        if not attr.isidentifier():
            raise ValueError(f"Invalid attribute name '{attr}' in '{{{field_name}}}'")

        if attr in restricted_attrs:
            raise ValueError(f"Restricted placeholder '{{{field_name}}}'")

        if root == "performance" and attr not in performance_allowed_attrs:
            raise ValueError(f"Unknown performance attribute '{attr}' in '{{{field_name}}}'")

    return True
