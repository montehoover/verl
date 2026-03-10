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


def build_worker_list(team_name: str) -> str:
    """
    Generate a newline-separated list of employees with positions for a given department.

    Args:
        team_name: The department name.

    Returns:
        A newline-separated string where each line is formatted as
        "Name: {employee.name}, Position: {employee.position}" if any employees exist,
        otherwise the string "No employees found."

    Raises:
        KeyError: If the department name is invalid (not known).
    """
    if team_name not in DEPARTMENTS:
        raise KeyError(f"Invalid department name: {team_name}")

    employees = DEPARTMENTS[team_name]
    if not employees:
        return "No employees found."
    return "\n".join(
        f"Name: {employee['name']}, Position: {employee['position']}" for employee in employees
    )
