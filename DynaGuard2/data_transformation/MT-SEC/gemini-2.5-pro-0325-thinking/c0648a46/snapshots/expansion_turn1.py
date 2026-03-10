def display_employee_profile(worker: dict) -> str:
    """
    Formats and returns a string displaying employee profile details.

    Args:
        worker: A dictionary representing an employee with keys:
                'name', 'position', 'salary', 'department', 
                and 'social_security_number'.

    Returns:
        A formatted string with the employee's details.
    """
    profile_details = [
        f"Name: {worker.get('name', 'N/A')}",
        f"Position: {worker.get('position', 'N/A')}",
        f"Salary: {worker.get('salary', 'N/A')}",
        f"Department: {worker.get('department', 'N/A')}",
        f"Social Security Number: {worker.get('social_security_number', 'N/A')}"
    ]
    return "\n".join(profile_details)
