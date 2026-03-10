import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EDITABLE_FIELDS = {"name", "position", "department"}

def modify_employee_profile(worker: dict, changes: dict, is_privileged: bool) -> dict:
    """
    Modifies certain fields within an employee's profile.

    Args:
        worker: dict, represents the current employee profile.
        changes: dict, a dictionary specifying the fields and new values.
        is_privileged: bool, indicates if the operation is by an admin.

    Returns:
        dict: The revised employee profile dictionary.
    """
    # Assuming 'worker' has an identifier, e.g., 'employee_id'. If not, this log might be less specific.
    # For this example, let's assume 'name' can serve as a temporary identifier for logging if 'employee_id' isn't present.
    worker_identifier = worker.get('employee_id', worker.get('name', 'Unknown Employee'))

    for field, new_value in changes.items():
        old_value = worker.get(field, 'N/A')
        log_prefix = f"Employee ID: {worker_identifier} - User (Privileged: {is_privileged})"

        # Guard clause: if not privileged and field is not editable, log and skip update
        if not is_privileged and field not in EDITABLE_FIELDS:
            logging.warning(f"{log_prefix} - Attempt to change non-editable field '{field}' to '{new_value}' DENIED.")
            continue

        if old_value == new_value:
            logging.info(f"{log_prefix} - Field '{field}' already set to '{new_value}'. No change made.")
        else:
            worker[field] = new_value
            logging.info(f"{log_prefix} - Successfully changed field '{field}' from '{old_value}' to '{new_value}'.")
            
    return worker
