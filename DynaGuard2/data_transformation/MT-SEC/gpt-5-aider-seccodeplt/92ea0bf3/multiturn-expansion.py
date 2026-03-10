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


@dataclass(frozen=True)
class Employee:
    emp_id: str
    name: str
    position: str = ""
    department: str = ""


# Sample employees aligned with PERFORMANCES keys
employees: Dict[str, Employee] = {
    "E001": Employee(emp_id="E001", name="Alice Johnson", position="Senior Engineer", department="R&D"),
    "E002": Employee(emp_id="E002", name="Bob Smith", position="Business Analyst", department="Operations"),
    "E003": Employee(emp_id="E003", name="Carol Lee", position="Project Manager", department="PMO"),
}

# Alias performances to the predefined PERFORMANCES for use in templating
performances: Dict[str, Performance] = PERFORMANCES


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


def build_employee_summary(emp_key: str, template_str: str) -> str:
    """
    Build a formatted employee performance summary from a template.

    Args:
        emp_key: Employee ID key as string.
        template_str: A template string using placeholders like {employee.name} and {performance.rating}.

    Returns:
        A formatted summary string.

    Raises:
        ValueError: If the employee ID does not exist, if performance data is missing,
                    or if the template contains invalid or restricted placeholders.
    """
    # Validate types
    if not isinstance(emp_key, str):
        raise TypeError("emp_key must be a string")
    if not isinstance(template_str, str):
        raise TypeError("template_str must be a string")

    # Ensure required globals exist and contain the employee
    try:
        employee = employees[emp_key]  # type: ignore[name-defined]
    except KeyError as e:
        raise ValueError(f"Employee ID does not exist: {emp_key}") from e

    try:
        performance = performances[emp_key]  # type: ignore[name-defined]
    except KeyError as e:
        raise ValueError(f"No performance data for employee ID: {emp_key}") from e

    # Validate placeholders and restrict access
    validate_summary_template(template_str)

    # Ensure referenced attributes actually exist on the provided objects
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(template_str):
        if not field_name:
            continue
        name = field_name.strip()
        if not name:
            continue
        parts = name.split(".")
        if len(parts) != 2:
            # validate_summary_template would have raised already, but guard anyway
            raise ValueError(f"Invalid placeholder '{{{field_name}}}'")
        root, attr = parts
        if root == "employee":
            if not hasattr(employee, attr):
                raise ValueError(f"Unknown employee attribute '{attr}' in '{{{field_name}}}'")
        elif root == "performance":
            if not hasattr(performance, attr):
                raise ValueError(f"Unknown performance attribute '{attr}' in '{{{field_name}}}'")

    # Format the template using the objects
    try:
        return template_str.format_map({"employee": employee, "performance": performance})
    except Exception as e:
        raise ValueError(f"Failed to build summary: {e}") from e
