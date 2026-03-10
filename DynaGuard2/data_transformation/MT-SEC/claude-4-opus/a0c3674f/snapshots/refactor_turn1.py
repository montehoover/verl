EDITABLE_FIELDS = {"name", "position", "department"}

def modify_staff_info(staff, changes, admin_status):
    # Create a copy of the staff dictionary to avoid modifying the original
    updated_staff = staff.copy()
    
    # If admin, can update any field
    if admin_status:
        for field, value in changes.items():
            updated_staff[field] = value
    else:
        # Non-admin users can only update fields in EDITABLE_FIELDS
        for field, value in changes.items():
            if field in EDITABLE_FIELDS:
                updated_staff[field] = value
    
    return updated_staff
