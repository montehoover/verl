EDITABLE_FIELDS = {"name", "position", "department"}


def filter_editable_changes(changes, editable_fields):
    """Filter changes to only include editable fields.
    
    Args:
        changes: Dictionary containing field-value pairs for updates.
        editable_fields: Set of field names that are allowed to be edited.
        
    Returns:
        Dictionary containing only the editable field updates.
    """
    return {field: value for field, value in changes.items() 
            if field in editable_fields}


def apply_changes(staff, changes):
    """Apply changes to a staff dictionary.
    
    Args:
        staff: Dictionary containing current staff information.
        changes: Dictionary containing field-value pairs for updates.
        
    Returns:
        New dictionary with applied changes.
    """
    updated_staff = staff.copy()
    updated_staff.update(changes)
    return updated_staff


def modify_staff_info(staff, changes, admin_status):
    """Modify staff information based on user permissions.
    
    Args:
        staff: Dictionary containing current staff information.
        changes: Dictionary containing field-value pairs for updates.
        admin_status: Boolean indicating if user has admin privileges.
        
    Returns:
        Updated staff dictionary with valid changes applied.
    """
    # Guard clause: Admins can update any field
    if admin_status:
        return apply_changes(staff, changes)
    
    # Non-admins can only update editable fields
    valid_changes = filter_editable_changes(changes, EDITABLE_FIELDS)
    
    return apply_changes(staff, valid_changes)
