EMPLOYEES_BY_DEPARTMENT = {
    "Engineering": [
        {"name": "Alice", "position": "Senior Engineer"},
        {"name": "Bob", "position": "Software Engineer"},
        {"name": "Charlie", "position": "Intern"}
    ],
    "Sales": [
        {"name": "David", "position": "Sales Manager"},
        {"name": "Eve", "position": "Sales Representative"}
    ],
    "HR": [],
    "Marketing": [
        {"name": "Frank", "position": "Marketing Lead"},
        {"name": "Grace", "position": "Content Creator"},
        {"name": "Heidi", "position": "Social Media Manager"}
    ]
}

def build_worker_list(team_name: str) -> str:
    """
    Generates a listing of employee names and positions for a given department.

    Args:
        team_name: The name of the department.

    Returns:
        A newline-separated list of employee names and positions in the format
        "Name: {employee.name}, Position: {employee.position}".
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
    
    worker_details = [
        f"Name: {employee['name']}, Position: {employee['position']}" 
        for employee in employees
    ]
    return "\n".join(worker_details)
