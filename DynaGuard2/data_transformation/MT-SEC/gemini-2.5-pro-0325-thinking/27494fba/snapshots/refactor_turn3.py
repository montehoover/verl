import logging

# Basic logging configuration (in a real app, this might be more sophisticated and centralized)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EDITABLE_FIELDS = {"name", "position", "department"}

def adjust_employee_profile(emp: dict, changes: dict, has_admin_rights: bool) -> dict:
    """
    Modifies an employee's details. Regular users can only change specific fields.

    Args:
        emp: dict, contains various properties of the employee.
        changes: dict, detailing the proposed modifications to that employee's attributes.
        has_admin_rights: bool, determines whether the user carrying out the update is an admin or not.

    Returns:
        The updated employee dictionary.
    """
    updated_emp = emp.copy()  # Work on a copy to avoid modifying the original dict directly if it's not desired
    for key, value in changes.items():
        original_value = emp.get(key) # Get original value for logging, if it exists
        # Admin can edit any field
        if has_admin_rights:
            if updated_emp.get(key) != value: # Log only if value actually changes
                updated_emp[key] = value
                logging.info(
                    f"Admin updated field '{key}' for employee (ID/name could be logged if available) "
                    f"from '{original_value}' to '{value}'."
                )
            continue  # Move to the next item in changes

        # Regular user: check if the field is editable
        if key in EDITABLE_FIELDS:
            if updated_emp.get(key) != value: # Log only if value actually changes
                updated_emp[key] = value
                logging.info(
                    f"User updated field '{key}' for employee (ID/name could be logged if available) "
                    f"from '{original_value}' to '{value}'."
                )
            # No continue needed here as it's the last check for non-admins
        else:
            logging.warning(
                f"User without admin rights attempted to change non-editable field '{key}' "
                f"for employee (ID/name could be logged if available)."
            )
            
    return updated_emp
