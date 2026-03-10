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


def update_and_log_profile(profile: dict, modifications: dict) -> tuple[dict, list[str]]:
    """
    Updates an employee's profile with new values and logs the changes.

    Args:
        profile: The original employee profile dictionary.
        modifications: A dictionary containing the changes to apply.
                       Keys are the profile attributes to update, and values
                       are the new values.

    Returns:
        A tuple containing:
            - The updated profile dictionary.
            - A list of strings, where each string is a log of a change made.
    """
    change_log = []
    updated_profile = profile.copy()  # Work on a copy to avoid modifying the original dict directly if passed by reference

    for key, new_value in modifications.items():
        old_value = updated_profile.get(key)
        if old_value != new_value:
            if old_value is None:
                change_log.append(f"Added '{key}' with value '{new_value}'.")
            else:
                change_log.append(f"Updated '{key}' from '{old_value}' to '{new_value}'.")
            updated_profile[key] = new_value
        elif key not in updated_profile: # Handles case where key is new and new_value is None (though less common for updates)
            change_log.append(f"Added '{key}' with value '{new_value}'.")
            updated_profile[key] = new_value


    return updated_profile, change_log

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

    # Example Usage for update_and_log_profile
    print("\n--- Updating Employee 1 Profile ---")
    employee1_modifications = {
        "salary": "$95,000",
        "position": "Senior Software Engineer",
        "email": "alice.wonderland@example.com" # New field
    }
    updated_employee1_profile, employee1_changelog = update_and_log_profile(employee1_profile, employee1_modifications)

    print("Updated Profile:")
    print(display_employee_profile(updated_employee1_profile))
    print("\nChange Log:")
    for log_entry in employee1_changelog:
        print(log_entry)

    print("\n--- Original Employee 1 Profile (should be unchanged if update_and_log_profile worked on a copy) ---")
    print(display_employee_profile(employee1_profile))


    print("\n--- Updating Employee 2 Profile (no actual changes) ---")
    employee2_no_changes = {
        "salary": "$110,000" # Same as original
    }
    updated_employee2_profile, employee2_changelog = update_and_log_profile(employee2_profile, employee2_no_changes)
    print("Updated Profile:")
    print(display_employee_profile(updated_employee2_profile))
    print("\nChange Log (should be empty):")
    for log_entry in employee2_changelog:
        print(log_entry)
    if not employee2_changelog:
        print("No changes were made.")
