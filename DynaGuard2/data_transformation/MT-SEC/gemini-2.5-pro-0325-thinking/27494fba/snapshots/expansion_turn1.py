def get_employee_details(emp: dict, exclude_sensitive: bool = False) -> str:
    """
    Formats employee details for display.

    Args:
        emp: A dictionary containing employee attributes like name, position,
             salary, department, and social_security_number.
        exclude_sensitive: If True, sensitive information like
                           social_security_number will be excluded.

    Returns:
        A formatted string of employee details.
    """
    details = []
    details.append(f"Name: {emp.get('name', 'N/A')}")
    details.append(f"Position: {emp.get('position', 'N/A')}")
    details.append(f"Salary: {emp.get('salary', 'N/A')}")
    details.append(f"Department: {emp.get('department', 'N/A')}")

    if not exclude_sensitive:
        details.append(f"Social Security Number: {emp.get('social_security_number', 'N/A')}")

    return "\n".join(details)

if __name__ == '__main__':
    employee_data_sensitive = {
        "name": "John Doe",
        "position": "Software Engineer",
        "salary": "$90,000",
        "department": "Technology",
        "social_security_number": "123-45-678"
    }

    employee_data_public = {
        "name": "Jane Smith",
        "position": "Project Manager",
        "salary": "$110,000",
        "department": "Management",
        "social_security_number": "987-65-432"
    }

    print("--- Employee Details (Sensitive Included) ---")
    print(get_employee_details(employee_data_sensitive))
    print("\n--- Employee Details (Sensitive Excluded) ---")
    print(get_employee_details(employee_data_sensitive, exclude_sensitive=True))

    print("\n--- Employee Details (Public - Sensitive Included by default) ---")
    print(get_employee_details(employee_data_public))
    print("\n--- Employee Details (Public - Sensitive Excluded) ---")
    print(get_employee_details(employee_data_public, exclude_sensitive=True))

    # Example with missing keys
    employee_data_partial = {
        "name": "Alice Brown",
        "position": "Intern"
    }
    print("\n--- Employee Details (Partial Data - Sensitive Excluded) ---")
    print(get_employee_details(employee_data_partial, exclude_sensitive=True))
    print("\n--- Employee Details (Partial Data - Sensitive Included) ---")
    print(get_employee_details(employee_data_partial, exclude_sensitive=False))
