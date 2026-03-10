EDITABLE_FIELDS = {"name", "position", "department"}

def can_edit_field(field, admin_flag):
    """Check if a field can be edited based on admin status."""
    if admin_flag:
        return True
    return field in EDITABLE_FIELDS

def edit_personnel_info(employee_record, adjustments, admin_flag):
    # Create a copy of the employee record to avoid modifying the original
    updated_record = employee_record.copy()
    
    # Apply the adjustments for fields that can be edited
    for field, value in adjustments.items():
        if can_edit_field(field, admin_flag):
            updated_record[field] = value
    
    return updated_record
