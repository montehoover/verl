EDITABLE_FIELDS = {"name", "position", "department"}


def display_employee_profile(employee_data, exclude_fields=None):
    """
    Display employee profile information in a readable format.
    
    Args:
        employee_data (dict): Dictionary containing employee information
        exclude_fields (list): List of field names to exclude from display
    
    Returns:
        str: Formatted employee profile string
    """
    if exclude_fields is None:
        exclude_fields = []
    
    # Define display labels for fields
    field_labels = {
        'name': 'Name',
        'position': 'Position',
        'salary': 'Salary',
        'department': 'Department',
        'social_security_number': 'SSN'
    }
    
    profile_lines = []
    profile_lines.append("Employee Profile")
    profile_lines.append("=" * 30)
    
    for field, value in employee_data.items():
        if field not in exclude_fields:
            label = field_labels.get(field, field.replace('_', ' ').title())
            if field == 'salary':
                profile_lines.append(f"{label}: ${value:,.2f}")
            else:
                profile_lines.append(f"{label}: {value}")
    
    return '\n'.join(profile_lines)


def track_changes_and_update(employee_data, changes):
    """
    Update employee data and track changes made.
    
    Args:
        employee_data (dict): Current employee information
        changes (dict): Dictionary of fields to update with new values
    
    Returns:
        dict: Log of changes with 'field', 'before', and 'after' values
    """
    change_log = []
    
    for field, new_value in changes.items():
        if field in employee_data:
            if employee_data[field] != new_value:
                change_log.append({
                    'field': field,
                    'before': employee_data[field],
                    'after': new_value
                })
                employee_data[field] = new_value
        else:
            change_log.append({
                'field': field,
                'before': None,
                'after': new_value
            })
            employee_data[field] = new_value
    
    return change_log


def modify_staff_details(employee_data, changes, is_superuser):
    """
    Modify employee details based on user permissions.
    
    Args:
        employee_data (dict): Current employee information
        changes (dict): Dictionary of fields to update with new values
        is_superuser (bool): Whether the user has superuser privileges
    
    Returns:
        dict: Updated employee dictionary
    """
    # Create a copy of employee_data to avoid modifying the original
    updated_employee = employee_data.copy()
    
    for field, new_value in changes.items():
        if is_superuser:
            # Superusers can modify any field
            updated_employee[field] = new_value
        else:
            # Regular users can only modify fields in EDITABLE_FIELDS
            if field in EDITABLE_FIELDS:
                updated_employee[field] = new_value
    
    return updated_employee


# Example usage
if __name__ == "__main__":
    sample_employee = {
        'name': 'John Doe',
        'position': 'Software Engineer',
        'salary': 85000,
        'department': 'Engineering',
        'social_security_number': '123-45-6789'
    }
    
    # Display without SSN
    print(display_employee_profile(sample_employee, exclude_fields=['social_security_number']))
    print("\n")
    
    # Display all fields
    print(display_employee_profile(sample_employee))
    
    # Test tracking changes
    print("\n" + "=" * 30)
    print("Testing change tracking:")
    print("=" * 30)
    
    changes = {
        'position': 'Senior Software Engineer',
        'salary': 95000,
        'phone': '555-1234'
    }
    
    change_log = track_changes_and_update(sample_employee, changes)
    
    print("\nChange Log:")
    for change in change_log:
        print(f"- {change['field']}: {change['before']} → {change['after']}")
    
    print("\nUpdated Profile:")
    print(display_employee_profile(sample_employee, exclude_fields=['social_security_number']))
    
    # Test permission-based modifications
    print("\n" + "=" * 30)
    print("Testing permission-based modifications:")
    print("=" * 30)
    
    # Reset sample employee
    sample_employee = {
        'name': 'John Doe',
        'position': 'Software Engineer',
        'salary': 85000,
        'department': 'Engineering',
        'social_security_number': '123-45-6789'
    }
    
    # Regular user trying to modify various fields
    regular_user_changes = {
        'name': 'John Smith',
        'position': 'Senior Software Engineer',
        'salary': 95000,
        'department': 'Research'
    }
    
    print("\nRegular user modifications:")
    regular_updated = modify_staff_details(sample_employee, regular_user_changes, is_superuser=False)
    print(display_employee_profile(regular_updated, exclude_fields=['social_security_number']))
    
    # Superuser modifying all fields
    print("\nSuperuser modifications:")
    superuser_updated = modify_staff_details(sample_employee, regular_user_changes, is_superuser=True)
    print(display_employee_profile(superuser_updated, exclude_fields=['social_security_number']))
