EDITABLE_FIELDS = {"name", "position", "department"}

def validate_field_access(field, is_superuser):
    """Check if a field can be modified based on user permissions."""
    return is_superuser or field in EDITABLE_FIELDS

def filter_valid_changes(changes, is_superuser):
    """Filter changes to only include fields the user has permission to modify."""
    return {
        field: value 
        for field, value in changes.items() 
        if validate_field_access(field, is_superuser)
    }

def apply_changes(emp_data, valid_changes):
    """Apply validated changes to employee data."""
    updated_employee = emp_data.copy()
    updated_employee.update(valid_changes)
    return updated_employee

def alter_employee_details(emp_data, changes, is_superuser):
    valid_changes = filter_valid_changes(changes, is_superuser)
    return apply_changes(emp_data, valid_changes)
