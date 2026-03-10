EDITABLE_FIELDS = {"name", "position", "department"}

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


def edit_profile_data(profile: dict, modifications: dict, is_superuser: bool) -> dict:
    """
    Edits an employee's profile based on user permissions.

    Args:
        profile: The original employee profile dictionary.
        modifications: A dictionary containing the changes to apply.
        is_superuser: A boolean indicating if the user has superuser privileges.

    Returns:
        The updated profile dictionary.
    """
    updated_profile = profile.copy()

    for key, new_value in modifications.items():
        if is_superuser or key in EDITABLE_FIELDS:
            updated_profile[key] = new_value
        else:
            print(f"Warning: User does not have permission to edit '{key}'. Skipping update for this field.")
    return updated_profile

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

    # Example Usage for edit_profile_data
    print("\n--- Editing Employee 1 Profile (as Superuser) ---")
    superuser_modifications = {
        "salary": "$100,000",  # Superuser can change salary
        "social_security_number": "XXX-XX-6789", # Superuser can change SSN
        "department": "Advanced Technology"
    }
    # Use updated_employee1_profile from previous step as the base for this edit
    edited_profile_superuser = edit_profile_data(updated_employee1_profile, superuser_modifications, is_superuser=True)
    print("Profile after superuser edit:")
    print(display_employee_profile(edited_profile_superuser))

    print("\n--- Editing Employee 1 Profile (as Non-Superuser) ---")
    non_superuser_modifications = {
        "salary": "$120,000",  # Non-superuser cannot change salary
        "position": "Lead Software Engineer", # Non-superuser can change position
        "department": "Core Technology" # Non-superuser can change department
    }
    # Use edited_profile_superuser as the base for this edit
    edited_profile_non_superuser = edit_profile_data(edited_profile_superuser, non_superuser_modifications, is_superuser=False)
    print("Profile after non-superuser edit:")
    print(display_employee_profile(edited_profile_non_superuser))

    print("\n--- Editing Employee 3 Profile (as Non-Superuser, adding fields) ---")
    employee3_modifications_non_superuser = {
        "name": "Carol 'Captain Marvel' Danvers", # Can edit
        "position": "Avenger", # Can edit
        "salary": "$500,000", # Cannot edit
        "department": "Space Operations" # Can edit
    }
    edited_employee3_profile_non_superuser = edit_profile_data(employee3_profile, employee3_modifications_non_superuser, is_superuser=False)
    print("Employee 3 Profile after non-superuser edit:")
    print(display_employee_profile(edited_employee3_profile_non_superuser))

    print("\n--- Editing Employee 3 Profile (as Superuser, adding fields) ---")
    employee3_modifications_superuser = {
        "name": "Carol 'Captain Marvel' Danvers",
        "position": "Avenger",
        "salary": "$500,000", # Can edit
        "department": "Space Operations",
        "social_security_number": "N/A - Alien" # Can edit
    }
    # Use the original employee3_profile for a clean test
    edited_employee3_profile_superuser = edit_profile_data(employee3_profile, employee3_modifications_superuser, is_superuser=True)
    print("Employee 3 Profile after superuser edit:")
    print(display_employee_profile(edited_employee3_profile_superuser, exclude_sensitive=False))
