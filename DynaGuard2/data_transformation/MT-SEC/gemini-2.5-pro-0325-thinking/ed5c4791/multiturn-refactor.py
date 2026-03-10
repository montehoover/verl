import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EDITABLE_FIELDS = {"name", "position", "department"}

def _apply_modifications(worker_data: dict, alterations: dict, allowed_fields: set, is_admin: bool) -> dict:
    """
    Applies modifications to worker_data based on permissions and logs attempts.

    This is a pure function that takes employee data, proposed changes,
    a set of fields modifiable by non-admins, and an admin status flag.
    It iterates through each proposed alteration, logs the attempt,
    and applies the change if the user has the necessary permissions.

    Args:
        worker_data: A dictionary representing the employee's data.
                     This should be a copy of the original data to avoid side effects.
        alterations: A dictionary where keys are field names to be modified
                     and values are the new values for these fields.
        allowed_fields: A set of strings, where each string is a field name
                        that can be modified by a non-administrative user.
        is_admin: A boolean flag; True if the user performing the modification
                  has administrative privileges, False otherwise.

    Returns:
        A dictionary containing the updated employee data.
    """
    for key, value in alterations.items():
        log_message_prefix = (
            f"User (admin: {is_admin}) attempting to modify field '{key}'"
        )
        # Guard clause: if not admin and the key is not in allowed_fields, log and skip.
        if not is_admin and key not in allowed_fields:
            logger.info(f"{log_message_prefix}. DENIED: Insufficient privileges.")
            continue
        
        logger.info(f"{log_message_prefix} to '{value}'. ALLOWED.")
        worker_data[key] = value
    return worker_data

def modify_employee_data(worker: dict, alterations: dict, admin_privileges: bool) -> dict:
    """
    Modifies an employee's details, orchestrating the modification process.

    This function serves as the primary interface for modifying employee data.
    It ensures that modifications are made on a copy of the employee's data
    to prevent unintended side effects on the original data structure.
    It then delegates the actual modification logic, including permission checks
    and logging, to the `_apply_modifications` helper function.

    Regular users are allowed to change only specific fields predefined in
    the global `EDITABLE_FIELDS` set. Users with administrative privileges
    can modify any field.

    Args:
        worker: A dictionary containing the employee's current attributes
                (e.g., "name", "position", "salary").
        alterations: A dictionary detailing the proposed modifications.
                     Keys are the names of attributes to change, and values are
                     their new values.
        admin_privileges: A boolean indicating whether the user performing
                          the update has administrative rights. True for admins,
                          False for regular users.

    Returns:
        A new dictionary representing the employee's data after applying
        the permitted modifications.
    """
    updated_worker = worker.copy()  # Work on a copy to avoid modifying the original dict
    logger.info(
        f"Initiating modification attempt for employee data. Admin privileges: {admin_privileges}. "
        f"Attempted changes: {list(alterations.keys())}"
    )
    return _apply_modifications(updated_worker, alterations, EDITABLE_FIELDS, admin_privileges)
