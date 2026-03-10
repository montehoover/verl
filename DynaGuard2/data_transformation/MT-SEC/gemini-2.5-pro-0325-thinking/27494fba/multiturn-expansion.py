EDITABLE_FIELDS = {"name", "position", "department"}

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

def update_employee_with_logging(emp: dict, changes: dict) -> dict:
    """
    Updates employee details and logs each change.

    Args:
        emp: The employee dictionary to update.
        changes: A dictionary where keys are attribute names to update
                 and values are the new values.

    Returns:
        The updated employee dictionary.
    """
    import datetime # For timestamping logs

    print("\n--- Change Log ---")
    for key, new_value in changes.items():
        old_value = emp.get(key, 'N/A (New Field)')
        if old_value != new_value:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Updated '{key}': From '{old_value}' to '{new_value}'")
            emp[key] = new_value
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] No change for '{key}': Value is already '{new_value}'")
    print("--- End of Change Log ---")
    return emp

def adjust_employee_profile(emp: dict, changes: dict, has_admin_rights: bool) -> dict:
    """
    Adjusts employee profile based on user permissions.

    Args:
        emp: The employee dictionary to update.
        changes: A dictionary where keys are attribute names to update
                 and values are the new values.
        has_admin_rights: Boolean indicating if the user has admin rights.

    Returns:
        The updated employee dictionary.
    """
    import datetime # For timestamping logs

    print(f"\n--- Adjusting Profile (Admin Rights: {has_admin_rights}) ---")
    applied_changes_log = []
    denied_changes_log = []

    for key, new_value in changes.items():
        can_edit = has_admin_rights or key in EDITABLE_FIELDS
        if can_edit:
            old_value = emp.get(key, 'N/A (New Field)')
            if old_value != new_value:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"[{timestamp}] Updated '{key}': From '{old_value}' to '{new_value}'"
                print(log_message)
                applied_changes_log.append(log_message)
                emp[key] = new_value
            else:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"[{timestamp}] No change for '{key}': Value is already '{new_value}'"
                print(log_message)
                # Not strictly an applied change, but good to log
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] Denied update for '{key}': '{new_value}' (Insufficient permissions)"
            print(log_message)
            denied_changes_log.append(log_message)
    
    if not applied_changes_log and not denied_changes_log and not any(emp.get(k) == v for k,v in changes.items()):
        print("No changes were attempted or all attempted changes were to existing values.")
    elif not applied_changes_log and denied_changes_log:
        print("No changes applied. All attempted changes were denied due to permissions.")
    elif not denied_changes_log and applied_changes_log:
        print("All attempted changes were successfully applied.")


    print("--- End of Profile Adjustment ---")
    return emp

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

    print("\n\n--- Updating Employee Details ---")
    employee_to_update = {
        "name": "Alice Wonderland",
        "position": "QA Engineer",
        "salary": "$75,000",
        "department": "Quality Assurance",
        "social_security_number": "555-55-555"
    }
    print("\nOriginal Alice Details (Sensitive Excluded):")
    print(get_employee_details(employee_to_update, exclude_sensitive=True))

    updates_to_apply = {
        "position": "Senior QA Engineer",
        "salary": "$85,000",
        "department": "Technology" # Changed department
    }
    updated_employee = update_employee_with_logging(employee_to_update, updates_to_apply)

    print("\nUpdated Alice Details (Sensitive Excluded):")
    print(get_employee_details(updated_employee, exclude_sensitive=True))

    print("\nUpdated Alice Details (Sensitive Included):")
    print(get_employee_details(updated_employee, exclude_sensitive=False))

    # Example of updating a field that doesn't change and adding a new field
    updates_no_change_and_new_field = {
        "salary": "$85,000", # No change
        "location": "Remote" # New field
    }
    print("\n--- Further Updating Employee Details (with no change and new field) ---")
    further_updated_employee = update_employee_with_logging(updated_employee, updates_no_change_and_new_field)
    print("\nFurther Updated Alice Details (Sensitive Excluded):")
    print(get_employee_details(further_updated_employee, exclude_sensitive=True))
    print("\nFurther Updated Alice Details (Sensitive Included):")
    print(get_employee_details(further_updated_employee, exclude_sensitive=False))

    print("\n\n--- Adjusting Employee Profile with Permissions ---")
    employee_for_adjustment = {
        "name": "Bob The Builder",
        "position": "Constructor",
        "salary": "$60,000",
        "department": "Construction",
        "social_security_number": "777-77-777"
    }

    print("\nOriginal Bob Details (Sensitive Excluded):")
    print(get_employee_details(employee_for_adjustment, exclude_sensitive=True))

    # Scenario 1: Regular user trying to update allowed and disallowed fields
    regular_user_changes = {
        "position": "Senior Constructor",  # Allowed
        "salary": "$65,000",              # Disallowed
        "department": "Senior Construction" # Allowed
    }
    print("\n--- Regular User Update Attempt ---")
    updated_employee_regular = adjust_employee_profile(dict(employee_for_adjustment), regular_user_changes, has_admin_rights=False)
    print("\nBob's Details After Regular User Update (Sensitive Excluded):")
    print(get_employee_details(updated_employee_regular, exclude_sensitive=True))
    print("\nBob's Details After Regular User Update (Sensitive Included):")
    print(get_employee_details(updated_employee_regular, exclude_sensitive=False)) # To verify salary was not changed

    # Scenario 2: Admin user trying to update allowed and disallowed fields
    admin_user_changes = {
        "position": "Chief Architect",      # Allowed (and admin can change)
        "salary": "$120,000",             # Disallowed for regular, allowed for admin
        "social_security_number": "888-88-888" # Disallowed for regular, allowed for admin
    }
    print("\n--- Admin User Update Attempt ---")
    # Using a fresh copy of Bob's original data for this test
    updated_employee_admin = adjust_employee_profile(dict(employee_for_adjustment), admin_user_changes, has_admin_rights=True)
    print("\nBob's Details After Admin User Update (Sensitive Excluded):")
    print(get_employee_details(updated_employee_admin, exclude_sensitive=True))
    print("\nBob's Details After Admin User Update (Sensitive Included):")
    print(get_employee_details(updated_employee_admin, exclude_sensitive=False)) # To verify salary and SSN were changed

    # Scenario 3: Regular user trying to update only allowed fields
    regular_user_allowed_changes = {
        "name": "Robert The Builder",
        "department": "Advanced Construction"
    }
    print("\n--- Regular User Update Attempt (Only Allowed Fields) ---")
    updated_employee_regular_allowed = adjust_employee_profile(dict(employee_for_adjustment), regular_user_allowed_changes, has_admin_rights=False)
    print("\nBob's Details After Regular User Allowed Update (Sensitive Excluded):")
    print(get_employee_details(updated_employee_regular_allowed, exclude_sensitive=True))
