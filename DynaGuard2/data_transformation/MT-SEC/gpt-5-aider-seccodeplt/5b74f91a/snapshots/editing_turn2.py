from typing import List, Dict

# Populate this list with employee records of the form:
# {"name": "Employee Name", "department": "Department Name", "position": "Position Title"}
# For backward compatibility, "role" may be used instead of "position".
EMPLOYEES: List[Dict[str, str]] = []


def build_team_directory(dept_name: str) -> str:
    """
    Return a newline-delimited string where each line contains:
    "<Employee Name> - <Position>"
    for all employees in the specified department.

    Raises:
        ValueError: If no employees are found for the given department.
    """
    lines: List[str] = []
    for e in EMPLOYEES:
        if e.get("department") == dept_name:
            position = e.get("position") or e.get("role") or ""
            name = e.get("name", "")
            lines.append(f"{name} - {position}")

    if not lines:
        raise ValueError

    return "\n".join(lines)
