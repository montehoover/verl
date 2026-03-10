EDITABLE_FIELDS = {"name", "position", "department"}

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


def modify_staff_details(employee_data, changes, is_superuser):
    """
    Modifies employee details based on user permissions.

    Args:
        employee_data (dict): The current employee data.
        changes (dict): A dictionary containing the fields to update and their new values.
        is_superuser (bool): True if the user has superuser privileges, False otherwise.

    Returns:
        dict: The updated employee data dictionary.
              Returns the original employee_data if no authorized changes were made.
    """
    updated_data = employee_data.copy()  # Work on a copy to avoid modifying the original dict directly if no changes are made
    made_change = False
    for field, new_value in changes.items():
        if is_superuser or field in EDITABLE_FIELDS:
            if field in updated_data and updated_data[field] == new_value:
                # No actual change in value, skip updating and logging
                continue
            updated_data[field] = new_value
            made_change = True
            print(f"Field '{field}' updated to '{new_value}'.")
        else:
            print(f"User does not have permission to modify field '{field}'.")
    
    if made_change:
        return updated_data
    return employee_data # Return original if no authorized changes were made or no values actually changed


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

    # Example Usage for modify_staff_details
    hr_staff_member = {
        'name': 'Alice Wonderland',
        'position': 'HR Specialist',
        'salary': 70000,
        'department': 'Human Resources',
        'social_security_number': '***-**-7890',
        'employee_id': 'E1002'
    }

    print("\n--- Initial HR Staff Member Profile ---")
    print(display_employee_profile(hr_staff_member))

    # Regular user attempts to change name (allowed) and salary (not allowed)
    changes_by_regular_user = {
        'name': 'Alice Kingsleigh',
        'salary': 72000
    }
    print("\n--- Regular user attempting changes ---")
    updated_hr_staff_member_regular = modify_staff_details(hr_staff_member, changes_by_regular_user, is_superuser=False)
    print("\n--- HR Staff Member Profile after regular user attempt ---")
    print(display_employee_profile(updated_hr_staff_member_regular))

    # Superuser attempts to change name (allowed) and salary (allowed)
    changes_by_superuser = {
        'position': 'Senior HR Manager',
        'salary': 85000,
        'social_security_number': '***-**-NEWSSN' # Superuser can change sensitive info
    }
    print("\n--- Superuser attempting changes ---")
    # Apply changes to the record that was potentially modified by the regular user
    updated_hr_staff_member_superuser = modify_staff_details(updated_hr_staff_member_regular, changes_by_superuser, is_superuser=True)
    print("\n--- HR Staff Member Profile after superuser attempt ---")
    print(display_employee_profile(updated_hr_staff_member_superuser))

    # Attempt to change a field that doesn't exist (should be added if permissions allow)
    changes_new_field_superuser = {
        'office_location': 'Building A, Floor 3'
    }
    print("\n--- Superuser attempting to add a new field ---")
    updated_hr_staff_new_field = modify_staff_details(updated_hr_staff_member_superuser, changes_new_field_superuser, is_superuser=True)
    print("\n--- HR Staff Member Profile after superuser adds new field ---")
    print(display_employee_profile(updated_hr_staff_new_field))

    changes_new_field_regular = {
        'nickname': 'Ali' # Not in EDITABLE_FIELDS
    }
    print("\n--- Regular user attempting to add a new field (not in EDITABLE_FIELDS) ---")
    updated_hr_staff_new_field_regular = modify_staff_details(updated_hr_staff_new_field, changes_new_field_regular, is_superuser=False)
    print("\n--- HR Staff Member Profile after regular user attempts to add new field ---")
    print(display_employee_profile(updated_hr_staff_new_field_regular))

    changes_new_field_regular_editable = {
        'department': 'Global Human Resources' # 'department' is in EDITABLE_FIELDS
    }
    print("\n--- Regular user attempting to change an editable field (department) ---")
    updated_hr_staff_editable_field_regular = modify_staff_details(updated_hr_staff_new_field_regular, changes_new_field_regular_editable, is_superuser=False)
    print("\n--- HR Staff Member Profile after regular user changes an editable field ---")
    print(display_employee_profile(updated_hr_staff_editable_field_regular))
