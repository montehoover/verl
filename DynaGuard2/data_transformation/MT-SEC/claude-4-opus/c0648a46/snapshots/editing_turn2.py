def modify_employee_profile(worker: dict, changes: dict) -> dict:
    """
    Modify an employee's profile with data normalization.
    
    Args:
        worker: Current employee profile
        changes: Fields to be updated
        
    Returns:
        Updated profile with normalized data
    """
    updated_worker = worker.copy()
    
    for field, new_value in changes.items():
        if isinstance(new_value, str):
            # Strip whitespace and capitalize appropriately based on field
            normalized_value = new_value.strip()
            
            if field.lower() in ['name', 'first_name', 'last_name', 'full_name']:
                # Title case for names
                normalized_value = normalized_value.title()
            elif field.lower() in ['position', 'title', 'job_title', 'department']:
                # Title case for positions and departments
                normalized_value = normalized_value.title()
            elif field.lower() in ['email']:
                # Lowercase for emails
                normalized_value = normalized_value.lower()
            else:
                # Default: just strip whitespace
                normalized_value = normalized_value
                
            updated_worker[field] = normalized_value
        else:
            updated_worker[field] = new_value
    
    return updated_worker
