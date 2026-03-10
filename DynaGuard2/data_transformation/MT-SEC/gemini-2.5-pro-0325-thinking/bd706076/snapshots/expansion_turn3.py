EDITABLE_FIELDS = {"name", "position", "department"}

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


def modify_employee_with_logging(employee: dict, updates: dict) -> tuple[dict, list[str]]:
    """
    Modifies employee details and logs each change.

    Args:
        employee: The employee dictionary to be updated.
        updates: A dictionary containing the updates to apply.
                 Keys are the field names, and values are the new values.

    Returns:
        A tuple containing the updated employee dictionary and a list of log strings.
    """
    import datetime

    change_logs = []
    updated_employee = employee.copy()  # Work on a copy to avoid modifying the original dict directly if not intended

    for key, new_value in updates.items():
        old_value = updated_employee.get(key, 'N/A (New Field)')
        if old_value != new_value:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] Modified '{key}': from '{old_value}' to '{new_value}'"
            change_logs.append(log_entry)
            print(log_entry) # Or use a proper logging mechanism
            updated_employee[key] = new_value
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] No change for '{key}': value is already '{new_value}'"
            change_logs.append(log_entry)
            print(log_entry) # Or use a proper logging mechanism


    return updated_employee, change_logs


def update_employee_record(employee: dict, updates: dict, is_admin: bool) -> dict:
    """
    Updates employee records based on role-based permissions.

    Admin users can update any field. Regular users can only update fields
    specified in the EDITABLE_FIELDS set.

    Args:
        employee: The employee dictionary to be updated.
        updates: A dictionary containing the updates to apply.
        is_admin: A boolean indicating if the user has admin privileges.

    Returns:
        The updated employee dictionary.
    """
    updated_employee = employee.copy()

    for key, value in updates.items():
        if is_admin or key in EDITABLE_FIELDS:
            if updated_employee.get(key) != value:
                print(f"Updating '{key}' from '{updated_employee.get(key, 'N/A')}' to '{value}'")
                updated_employee[key] = value
            else:
                print(f"No change for '{key}': value is already '{value}'")
        else:
            print(f"Permission denied: Cannot update '{key}'. User is not an admin and field is not editable.")
    return updated_employee

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

    print("\n--- Modifying Employee Details with Logging ---")
    employee_to_modify = {
        "name": "Alice Wonderland",
        "position": "Lead Developer",
        "salary": "$120,000",
        "department": "Engineering",
        "social_security_number": "YYY-YY-YYYY"
    }
    updates_to_apply = {
        "position": "Senior Lead Developer",
        "salary": "$135,000",
        "email": "alice.wonderland@example.com" # New field
    }

    print(f"\nOriginal Employee Details for {employee_to_modify.get('name')}:")
    print(get_employee_details(employee_to_modify))

    updated_employee, logs = modify_employee_with_logging(employee_to_modify, updates_to_apply)

    print(f"\nUpdated Employee Details for {updated_employee.get('name')}:")
    print(get_employee_details(updated_employee))

    print("\nChange Log:")
    for log in logs:
        # Logs are already printed by the function, but we can iterate here if needed for other purposes
        pass # print(log) if not printed inside the function

    # Example: Modifying with no actual change
    print("\n--- Modifying Employee Details with No Actual Change ---")
    no_change_updates = {
        "position": "Senior Lead Developer" # Same as current
    }
    updated_employee_no_change, logs_no_change = modify_employee_with_logging(updated_employee, no_change_updates)
    print(f"\nEmployee Details for {updated_employee_no_change.get('name')} (after no change attempt):")
    print(get_employee_details(updated_employee_no_change))

    print("\n--- Updating Employee Record with Role-Based Permissions ---")
    employee_for_permission_test = {
        "name": "Bob The Builder",
        "position": "Constructor",
        "salary": "$70,000",
        "department": "Construction",
        "social_security_number": "ZZZ-ZZ-ZZZZ"
    }

    updates_by_admin = {
        "salary": "$75,000", # Admin can change salary
        "position": "Senior Constructor"
    }
    updates_by_user = {
        "position": "Lead Constructor", # User can change position
        "salary": "$80,000", # User cannot change salary
        "social_security_number": "NEW-SSN" # User cannot change SSN
    }

    print(f"\nOriginal details for {employee_for_permission_test.get('name')}:")
    print(get_employee_details(employee_for_permission_test))

    # Admin update
    print("\nAdmin performing updates:")
    updated_by_admin_employee = update_employee_record(employee_for_permission_test.copy(), updates_by_admin, is_admin=True)
    print(f"\nDetails for {updated_by_admin_employee.get('name')} after admin update:")
    print(get_employee_details(updated_by_admin_employee))

    # User update (on the original state for clarity)
    print("\nRegular user performing updates:")
    updated_by_user_employee = update_employee_record(employee_for_permission_test.copy(), updates_by_user, is_admin=False)
    print(f"\nDetails for {updated_by_user_employee.get('name')} after user update:")
    print(get_employee_details(updated_by_user_employee))

    # User update on already admin-updated record
    print("\nRegular user performing updates on admin-modified record:")
    # Let's try to update the record that admin already modified
    further_updates_by_user = {
        "name": "Robert The Builder", # User can change name
        "department": "Senior Construction Team" # User can change department
    }
    updated_further_by_user_employee = update_employee_record(updated_by_admin_employee.copy(), further_updates_by_user, is_admin=False)
    print(f"\nDetails for {updated_further_by_user_employee.get('name')} after further user update:")
    print(get_employee_details(updated_further_by_user_employee))
