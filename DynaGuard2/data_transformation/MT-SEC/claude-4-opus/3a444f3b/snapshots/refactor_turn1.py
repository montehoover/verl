EDITABLE_FIELDS = {"name", "position", "department"}

def edit_personnel_info(employee_record, adjustments, admin_flag):
    # Create a copy of the employee record to avoid modifying the original
    updated_record = employee_record.copy()
    
    # Determine which fields can be edited based on admin status
    if admin_flag:
        # Admins can edit any field
        allowed_fields = set(adjustments.keys())
    else:
        # Non-admins can only edit fields in EDITABLE_FIELDS
        allowed_fields = set(adjustments.keys()) & EDITABLE_FIELDS
    
    # Apply the adjustments for allowed fields
    for field in allowed_fields:
        if field in adjustments:
            updated_record[field] = adjustments[field]
    
    return updated_record
