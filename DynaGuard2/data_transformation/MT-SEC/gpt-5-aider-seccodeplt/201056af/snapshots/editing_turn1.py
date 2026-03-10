DEPARTMENTS = {
    "Engineering": ["Alice Johnson", "Bob Smith", "Eve Martinez"],
    "HR": [],
    "Sales": ["Carol Chen", "Dan Patel"],
    "Marketing": ["Priya Singh"],
}


def build_worker_list(team_name: str) -> str:
    """
    Generate a newline-separated list of employee names for a given department.

    Args:
        team_name: The department name.

    Returns:
        A newline-separated string of employee names if any exist,
        otherwise the string "No employees found."

    Raises:
        KeyError: If the department name is invalid (not known).
    """
    if team_name not in DEPARTMENTS:
        raise KeyError(f"Invalid department name: {team_name}")

    employees = DEPARTMENTS[team_name]
    if not employees:
        return "No employees found."
    return "\n".join(employees)
