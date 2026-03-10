def modify_employee_profile(worker: dict, changes: dict) -> tuple[dict, list[str]]:
    """
    Modifies an employee's profile and logs the changes.

    Args:
        worker: The current employee profile (dictionary).
        changes: A dictionary detailing the fields to be updated.

    Returns:
        A tuple containing the updated employee profile (dictionary)
        and a list of log entries (strings).
    """
    updated_worker = worker.copy()
    log_entries = []

    for key, value in changes.items():
        if key in updated_worker:
            if updated_worker[key] != value:
                log_entries.append(f"Field '{key}' changed from '{updated_worker[key]}' to '{value}'.")
            else:
                # Value is the same, could be logged as "attempted change" or ignored
                log_entries.append(f"Field '{key}' attempted update to same value '{value}'.")
        else:
            log_entries.append(f"Field '{key}' added with value '{value}'.")
        updated_worker[key] = value
    
    # Check for fields that might be in changes but not in worker initially
    # The above loop already handles adding new fields if they are in `changes`
    # and not in `worker`.

    return updated_worker, log_entries
