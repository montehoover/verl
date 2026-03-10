from string import Formatter
from typing import Any, Mapping, Set


ALLOWED_PERSONNEL_FIELDS: Set[str] = {"name", "position", "department"}
ALLOWED_REVIEW_FIELDS: Set[str] = {"rating"}
RESTRICTED_FIELDS: Set[str] = {"feedback", "bonus"}


def get_performance_record(employee_id: str, perf_map: Mapping[str, Any]) -> Any:
    """
    Pure function: retrieves the performance record for the given employee_id
    from the provided performance mapping.

    Raises:
        ValueError: If the employee ID is invalid or not present in the mapping.
    """
    try:
        return perf_map[employee_id]
    except Exception:
        raise ValueError("Invalid employee ID")


def format_performance_template(template: str, perf: Any) -> str:
    """
    Pure function: validates the template against allowed placeholders and formats it
    using the provided performance record.

    Raises:
        ValueError: If unauthorized or invalid placeholders are present, or if the template
                    formatting fails.
    """
    formatter = Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is None:
            continue

        segments = field_name.split(".")
        if any(seg in RESTRICTED_FIELDS for seg in segments):
            raise ValueError("Unauthorized access to restricted information")

        if segments[0] not in ("personnel", "review"):
            raise ValueError(f"Invalid placeholder: {field_name}")

        if len(segments) != 2:
            raise ValueError(f"Invalid placeholder: {field_name}")

        root, attr = segments
        if root == "personnel":
            if attr not in ALLOWED_PERSONNEL_FIELDS:
                raise ValueError(f"Invalid placeholder: {field_name}")
        elif root == "review":
            if attr not in ALLOWED_REVIEW_FIELDS:
                raise ValueError(f"Invalid placeholder: {field_name}")
        else:
            raise ValueError(f"Invalid placeholder: {field_name}")

    person = getattr(perf, "personnel", None)
    try:
        return template.format(personnel=person, review=perf)
    except Exception as exc:
        raise ValueError(f"Invalid template: {exc}") from exc


def compile_employee_evaluation(employee_id: str, template: str) -> str:
    """
    Compiles a performance summary for an employee by replacing placeholders in the template
    with the corresponding values from the employee and performance records.

    Args:
        employee_id: The unique ID of the employee.
        template: The template string which may contain placeholders like:
                  '{personnel.name}', '{personnel.position}', '{personnel.department}', '{review.rating}'.

    Returns:
        The formatted performance summary string.

    Raises:
        ValueError: If the employee ID is invalid, or if the template attempts to access unauthorized
                    information (e.g., feedback or bonus), or contains invalid placeholders.
    """
    perf_map = globals().get("performances")
    if not isinstance(perf_map, dict):
        raise ValueError("Invalid employee ID")

    perf = get_performance_record(employee_id, perf_map)
    return format_performance_template(template, perf)
