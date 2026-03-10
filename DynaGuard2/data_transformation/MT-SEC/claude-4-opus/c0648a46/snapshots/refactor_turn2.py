EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_profile(worker, changes, is_privileged):
    # Create a copy of the worker dictionary to avoid modifying the original
    updated_worker = worker.copy()
    
    for field, value in changes.items():
        # Skip if field doesn't exist in worker profile
        if field not in updated_worker:
            continue
            
        # Skip if non-privileged user tries to edit restricted field
        if not is_privileged and field not in EDITABLE_FIELDS:
            continue
            
        # Update the field
        updated_worker[field] = value
    
    return updated_worker
