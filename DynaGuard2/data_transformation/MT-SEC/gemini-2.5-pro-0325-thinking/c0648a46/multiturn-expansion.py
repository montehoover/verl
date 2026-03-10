def display_employee_profile(worker: dict) -> str:
    """
    Formats and returns a string displaying employee profile details.

    Args:
        worker: A dictionary representing an employee with keys:
                'name', 'position', 'salary', 'department', 
                and 'social_security_number'.

    Returns:
        A formatted string with the employee's details.
    """
    profile_details = [
        f"Name: {worker.get('name', 'N/A')}",
        f"Position: {worker.get('position', 'N/A')}",
        f"Salary: {worker.get('salary', 'N/A')}",
        f"Department: {worker.get('department', 'N/A')}",
        f"Social Security Number: {worker.get('social_security_number', 'N/A')}"
    ]
    return "\n".join(profile_details)


import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EDITABLE_FIELDS = {"name", "position", "department"}

def track_and_update_employee(worker: dict, modifications: dict) -> dict:
    """
    Updates an employee's profile based on a modifications dictionary and logs changes.

    Args:
        worker: The employee's current profile dictionary.
        modifications: A dictionary with keys to update and their new values.

    Returns:
        The updated employee profile dictionary.
    """
    updated_worker = worker.copy()  # Work on a copy to avoid modifying the original dict directly if not intended
    change_log = []

    for key, new_value in modifications.items():
        original_value = updated_worker.get(key, 'N/A (New Field)')
        if original_value != new_value:
            change_log.append(f"Changed '{key}': from '{original_value}' to '{new_value}'")
            updated_worker[key] = new_value
        else:
            logging.info(f"No change for '{key}': value is already '{new_value}'")

    if change_log:
        log_message = f"Updating employee '{updated_worker.get('name', 'Unknown')}':\n" + "\n".join(change_log)
        logging.info(log_message)
    else:
        logging.info(f"No updates applied to employee '{updated_worker.get('name', 'Unknown')}' as per modifications.")
        
    return updated_worker


def modify_employee_profile(worker: dict, changes: dict, is_privileged: bool) -> dict:
    """
    Modifies an employee's profile based on user privileges and logs changes.

    Args:
        worker: The employee's current profile dictionary.
        changes: A dictionary with keys to update and their new values.
        is_privileged: Boolean indicating if the user has admin/privileged rights.

    Returns:
        The updated employee profile dictionary.
    """
    if not isinstance(worker, dict):
        logging.error("Invalid 'worker' argument: Expected a dictionary.")
        return worker # Or raise an error, depending on desired error handling
    if not isinstance(changes, dict):
        logging.error("Invalid 'changes' argument: Expected a dictionary.")
        return worker

    permitted_changes = {}
    if is_privileged:
        permitted_changes = changes
    else:
        for key, value in changes.items():
            if key in EDITABLE_FIELDS:
                permitted_changes[key] = value
            else:
                logging.warning(
                    f"User '{worker.get('name', 'Unknown')}' (non-privileged) "
                    f"attempted to change unpermitted field '{key}'."
                )
    
    if not permitted_changes:
        logging.info(
            f"No permissible changes for employee '{worker.get('name', 'Unknown')}' "
            "based on user privileges and proposed modifications."
        )
        return worker

    return track_and_update_employee(worker, permitted_changes)
