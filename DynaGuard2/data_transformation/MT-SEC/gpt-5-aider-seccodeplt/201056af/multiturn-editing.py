import string
from types import SimpleNamespace
from typing import List

DEPARTMENTS = {
    "Engineering": [
        {"name": "Alice Johnson", "position": "Software Engineer"},
        {"name": "Bob Smith", "position": "DevOps Engineer"},
        {"name": "Eve Martinez", "position": "QA Engineer"},
    ],
    "HR": [],
    "Sales": [
        {"name": "Carol Chen", "position": "Account Executive"},
        {"name": "Dan Patel", "position": "Sales Manager"},
    ],
    "Marketing": [
        {"name": "Priya Singh", "position": "Marketing Specialist"},
    ],
}


def build_worker_list(team_name: str, list_template: str) -> str:
    """
    Generate a newline-separated, template-formatted list of employees for a given department.

    Args:
        team_name: The department name.
        list_template: A format template applied per employee. Supports placeholders like
                       '{employee.name}' and '{employee.position}'.

    Returns:
        A newline-separated string where each line is the template populated per employee.
        If no employees exist in the department, returns "No employees found."

    Raises:
        KeyError: If the department name is invalid (not known).
        ValueError: If the template attempts to access restricted fields (e.g., salary or personal email),
                    or if an unexpected error occurs during formatting.
    """
    if team_name not in DEPARTMENTS:
        raise KeyError(f"Invalid department name: {team_name}")

    employees: List[dict] = DEPARTMENTS[team_name]
    if not employees:
        return "No employees found."

    # Validate template fields for restricted access before formatting
    restricted_fields = {"salary", "personal_email", "personalEmail"}
    formatter = string.Formatter()
    try:
        for _, field_name, _, _ in formatter.parse(list_template):
            if not field_name:
                continue
            # Direct mention (unlikely but guarded)
            if field_name in restricted_fields or field_name.startswith("employee.") and field_name.split(".", 1)[1].split(".", 1)[0] in restricted_fields:
                raise ValueError(f"Access to restricted field in template: '{field_name}'")
    except Exception as e:
        # Any parsing anomaly should be surfaced as ValueError as per requirements
        raise ValueError(f"Template validation error: {e}") from e

    lines = []
    for emp in employees:
        employee_obj = SimpleNamespace(**emp)
        try:
            line = list_template.format(employee=employee_obj)
        except Exception as e:
            # Wrap any formatting errors as ValueError as requested
            raise ValueError(f"Template formatting error: {e}") from e
        lines.append(line)

    return "\n".join(lines)
