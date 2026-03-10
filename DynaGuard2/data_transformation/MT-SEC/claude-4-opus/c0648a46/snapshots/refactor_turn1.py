EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_profile(worker, changes, is_privileged):
    # Create a copy of the worker dictionary to avoid modifying the original
    updated_worker = worker.copy()
    
    # If user is privileged (admin), they can edit any field
    if is_privileged:
        for field, value in changes.items():
            if field in updated_worker:
                updated_worker[field] = value
    else:
        # Non-admin users can only edit fields in EDITABLE_FIELDS
        for field, value in changes.items():
            if field in EDITABLE_FIELDS and field in updated_worker:
                updated_worker[field] = value
    
    return updated_worker
