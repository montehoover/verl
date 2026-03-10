EMPLOYEES_BY_DEPARTMENT = {
    "Engineering": ["Alice", "Bob", "Charlie"],
    "Sales": ["David", "Eve"],
    "HR": [],
    "Marketing": ["Frank", "Grace", "Heidi"]
}

def build_worker_list(team_name: str) -> str:
    """
    Generates a simple listing of employee names for a given department.

    Args:
        team_name: The name of the department.

    Returns:
        A newline-separated list of employee names within that department.
        Returns "No employees found." if no employees are found for the
        specified department.

    Raises:
        KeyError: If the department name is invalid.
    """
    if team_name not in EMPLOYEES_BY_DEPARTMENT:
        raise KeyError(f"Invalid department name: {team_name}")

    employees = EMPLOYEES_BY_DEPARTMENT[team_name]

    if not employees:
        return "No employees found."
    
    return "\n".join(employees)
