EMPLOYEES_BY_DEPARTMENT = {
    "Engineering": [
        {"name": "Alice", "role": "Lead Engineer"},
        {"name": "Bob", "role": "Software Developer"},
        {"name": "Charlie", "role": "QA Engineer"}
    ],
    "Sales": [
        {"name": "David", "role": "Sales Manager"},
        {"name": "Eve", "role": "Sales Representative"}
    ],
    "HR": [
        {"name": "Frank", "role": "HR Director"},
        {"name": "Grace", "role": "HR Specialist"},
        {"name": "Heidi", "role": "Recruiter"}
    ],
    "Marketing": [] # Empty department for testing ValueError
}

def build_team_directory(dept_name: str) -> str:
    """
    Provides the names of all employees in a specified department.

    Args:
        dept_name: The name of the department.

    Returns:
        A string listing employees, with each employee's name followed by their role on the next line.

    Raises:
        ValueError: If no employees exist in the specified department or
                    if the department itself does not exist.
    """
    if dept_name not in EMPLOYEES_BY_DEPARTMENT:
        raise ValueError(f"Department '{dept_name}' not found.")

    employees = EMPLOYEES_BY_DEPARTMENT[dept_name]

    if not employees:
        raise ValueError(f"No employees found in department '{dept_name}'.")

    output_lines = []
    for employee_data in employees:
        output_lines.append(employee_data['name'])
        output_lines.append(employee_data['role'])
    return "\n".join(output_lines)

if __name__ == '__main__':
    # Example Usage
    try:
        print("--- Engineering Department ---")
        print(build_team_directory("Engineering"))
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print("\n--- Sales Department ---")
        print(build_team_directory("Sales"))
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print("\n--- NonExistent Department ---")
        print(build_team_directory("NonExistent"))
    except ValueError as e:
        print(f"Error: {e}")

    try:
        print("\n--- Marketing Department (Empty) ---")
        print(build_team_directory("Marketing"))
    except ValueError as e:
        print(f"Error: {e}")
