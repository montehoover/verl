def display_employee_profile(employee_data, exclude_fields=None):
    """
    Formats employee data into a readable string, excluding specified fields.

    Args:
        employee_data (dict): A dictionary containing employee information.
                              Example: {'name': 'John Doe', 'position': 'Developer', ...}
        exclude_fields (list, optional): A list of field names (keys) to exclude
                                         from the output. Defaults to None, which
                                         means no fields are excluded by default.
                                         Example: ['social_security_number']

    Returns:
        str: A formatted string representing the employee's profile.
    """
    if exclude_fields is None:
        exclude_fields = []

    profile_parts = []
    for key, value in employee_data.items():
        if key not in exclude_fields:
            # Replace underscores with spaces and capitalize words for better readability
            formatted_key = key.replace('_', ' ').title()
            profile_parts.append(f"{formatted_key}: {value}")

    return "\n".join(profile_parts)


def track_changes_and_update(employee_data, changes):
    """
    Updates employee details based on a changes dictionary and logs modifications.

    Args:
        employee_data (dict): The current employee data.
        changes (dict): A dictionary containing the fields to update and their new values.
                        Example: {'position': 'Senior Developer', 'salary': 125000}

    Returns:
        dict: A log of changes, where each key is the modified field and the value
              is a dictionary with 'old_value' and 'new_value'.
              Example: {'position': {'old_value': 'Developer', 'new_value': 'Senior Developer'}}
    """
    change_log = {}
    for key, new_value in changes.items():
        if key in employee_data:
            old_value = employee_data[key]
            if old_value != new_value:  # Log only if there's an actual change
                change_log[key] = {'old_value': old_value, 'new_value': new_value}
            employee_data[key] = new_value
        else:
            # If the key doesn't exist, it's a new addition
            change_log[key] = {'old_value': None, 'new_value': new_value}
            employee_data[key] = new_value
    return change_log

if __name__ == '__main__':
    # Example Usage for display_employee_profile
    sample_employee_data_display = {
        'name': 'Jane Doe',
        'position': 'Senior Software Engineer',
        'salary': 120000,
        'department': 'Technology',
        'email': 'jane.doe@example.com',
        'social_security_number': '***-**-1234',
        'employee_id': 'E1001'
    }

    print("--- Full Profile ---")
    print(display_employee_profile(sample_employee_data_display))
    print("\n--- Profile excluding SSN ---")
    print(display_employee_profile(sample_employee_data_display, exclude_fields=['social_security_number']))
    print("\n--- Profile excluding SSN and Salary ---")
    print(display_employee_profile(sample_employee_data_display, exclude_fields=['social_security_number', 'salary']))
    print("\n--- Profile with no exclusions (explicit empty list) ---")
    print(display_employee_profile(sample_employee_data_display, exclude_fields=[]))

    # Example Usage for track_changes_and_update
    employee_record = {
        'name': 'John Smith',
        'position': 'Junior Developer',
        'salary': 60000,
        'department': 'Engineering',
        'social_security_number': '***-**-5678'
    }

    print("\n--- Initial Employee Record ---")
    print(display_employee_profile(employee_record))

    changes_to_apply = {
        'position': 'Software Engineer',
        'salary': 75000,
        'email': 'john.smith@example.com' # New field
    }

    print(f"\n--- Applying Changes: {changes_to_apply} ---")
    change_log = track_changes_and_update(employee_record, changes_to_apply)

    print("\n--- Change Log ---")
    for field, values in change_log.items():
        print(f"Field: {field.replace('_', ' ').title()}")
        print(f"  Old Value: {values['old_value']}")
        print(f"  New Value: {values['new_value']}")

    print("\n--- Updated Employee Record ---")
    print(display_employee_profile(employee_record))

    # Example: No actual change in value
    no_real_changes = {'salary': 75000}
    print(f"\n--- Applying Changes with no real value difference: {no_real_changes} ---")
    change_log_no_diff = track_changes_and_update(employee_record, no_real_changes)
    print("\n--- Change Log (should be empty or minimal) ---")
    if not change_log_no_diff:
        print("No changes logged as values were the same.")
    else:
        for field, values in change_log_no_diff.items():
            print(f"Field: {field.replace('_', ' ').title()}")
            print(f"  Old Value: {values['old_value']}")
            print(f"  New Value: {values['new_value']}")
    print("\n--- Employee Record (should be unchanged from previous update) ---")
    print(display_employee_profile(employee_record))
