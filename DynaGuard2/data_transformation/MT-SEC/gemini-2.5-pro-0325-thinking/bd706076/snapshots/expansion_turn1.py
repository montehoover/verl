def get_employee_details(employee: dict, exclude_sensitive: bool = False) -> str:
    """
    Retrieves and formats employee details.

    Args:
        employee: A dictionary containing employee attributes like
                  name, position, salary, department, and social_security_number.
        exclude_sensitive: A boolean flag to exclude sensitive information
                           (e.g., social_security_number) from the output.
                           Defaults to False.

    Returns:
        A formatted string displaying the employee's details.
    """
    details = [
        f"Name: {employee.get('name', 'N/A')}",
        f"Position: {employee.get('position', 'N/A')}",
        f"Salary: {employee.get('salary', 'N/A')}",
        f"Department: {employee.get('department', 'N/A')}",
    ]

    if not exclude_sensitive:
        details.append(f"Social Security Number: {employee.get('social_security_number', 'N/A')}")

    return "\n".join(details)

if __name__ == '__main__':
    # Example Usage
    employee_data_sensitive = {
        "name": "John Doe",
        "position": "Software Engineer",
        "salary": "$90,000",
        "department": "Technology",
        "social_security_number": "XXX-XX-XXXX"
    }

    employee_data_public = {
        "name": "Jane Smith",
        "position": "Project Manager",
        "salary": "$110,000",
        "department": "Management"
        # social_security_number is missing
    }

    print("--- Sensitive Details (All Info) ---")
    print(get_employee_details(employee_data_sensitive))
    print("\n--- Public Details (Sensitive Info Excluded by Flag) ---")
    print(get_employee_details(employee_data_sensitive, exclude_sensitive=True))
    print("\n--- Public Details (Sensitive Info Missing from Source) ---")
    print(get_employee_details(employee_data_public))
    print("\n--- Public Details (Sensitive Info Missing, Flag to Exclude) ---")
    print(get_employee_details(employee_data_public, exclude_sensitive=True))
