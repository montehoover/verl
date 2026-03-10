def display_employee_profile(profile: dict, exclude_sensitive: bool = False) -> str:
    """
    Formats and displays an employee's profile.

    Args:
        profile: A dictionary containing employee details.
                 Expected keys: "name", "position", "salary", "department",
                                "social_security_number".
        exclude_sensitive: If True, excludes sensitive information like
                           social_security_number from the output.
                           Defaults to False.

    Returns:
        A formatted string representing the employee's profile.
    """
    display_items = [
        f"Name: {profile.get('name', 'N/A')}",
        f"Position: {profile.get('position', 'N/A')}",
        f"Salary: {profile.get('salary', 'N/A')}",
        f"Department: {profile.get('department', 'N/A')}",
    ]

    if not exclude_sensitive:
        display_items.append(f"Social Security Number: {profile.get('social_security_number', 'N/A')}")

    return "\n".join(display_items)

if __name__ == '__main__':
    # Example Usage
    employee1_profile = {
        "name": "Alice Wonderland",
        "position": "Software Engineer",
        "salary": "$90,000",
        "department": "Technology",
        "social_security_number": "XXX-XX-1234"
    }

    employee2_profile = {
        "name": "Bob The Builder",
        "position": "Project Manager",
        "salary": "$110,000",
        "department": "Construction",
        "social_security_number": "YYY-YY-5678"
    }

    print("--- Employee 1 Profile (Full Details) ---")
    print(display_employee_profile(employee1_profile))
    print("\n--- Employee 1 Profile (Sensitive Info Excluded) ---")
    print(display_employee_profile(employee1_profile, exclude_sensitive=True))

    print("\n--- Employee 2 Profile (Full Details, default exclude_sensitive=False) ---")
    print(display_employee_profile(employee2_profile))
    print("\n--- Employee 2 Profile (Sensitive Info Excluded) ---")
    print(display_employee_profile(employee2_profile, exclude_sensitive=True))

    # Example with missing data
    employee3_profile = {
        "name": "Carol Danvers",
        "position": "Captain"
        # Missing salary, department, ssn
    }
    print("\n--- Employee 3 Profile (Missing Data, Full Details) ---")
    print(display_employee_profile(employee3_profile))
    print("\n--- Employee 3 Profile (Missing Data, Sensitive Info Excluded) ---")
    print(display_employee_profile(employee3_profile, exclude_sensitive=True))
