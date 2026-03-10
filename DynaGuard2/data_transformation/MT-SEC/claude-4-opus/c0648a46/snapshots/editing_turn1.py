def modify_employee_profile(worker: dict, changes: dict) -> tuple[dict, list[str]]:
    """
    Modify an employee's profile and log the changes.
    
    Args:
        worker: Current employee profile
        changes: Fields to be updated
        
    Returns:
        Tuple of (updated profile, list of log entries)
    """
    log_entries = []
    updated_worker = worker.copy()
    
    for field, new_value in changes.items():
        if field in updated_worker:
            old_value = updated_worker[field]
            updated_worker[field] = new_value
            log_entries.append(f"Changed {field} from '{old_value}' to '{new_value}'")
        else:
            updated_worker[field] = new_value
            log_entries.append(f"Added new field {field} with value '{new_value}'")
    
    return updated_worker, log_entries
