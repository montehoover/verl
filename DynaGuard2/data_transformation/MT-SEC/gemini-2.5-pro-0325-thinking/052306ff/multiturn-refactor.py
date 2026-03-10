import logging
import json # For pretty printing dicts in logs

EDITABLE_FIELDS = {"name", "position", "department"}

# Configure logging
logging.basicConfig(filename='employee_updates.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def _validate_changes(changes: dict, is_superuser: bool, allowed_fields: set) -> dict:
    """
    Validates the changes based on user privileges.

    Args:
        changes: dict, a dictionary containing key-value pairs for fields and their new data.
        is_superuser: bool, a flag that indicates whether the person making the update is an admin or not.
        allowed_fields: set, a set of fields that non-admin users are allowed to edit.

    Returns:
        dict: A dictionary containing only the valid changes.
    """
    validated_changes = {}
    for field, new_value in changes.items():
        if is_superuser or field in allowed_fields:
            validated_changes[field] = new_value
    return validated_changes

def _apply_changes(emp_data: dict, validated_changes: dict) -> dict:
    """
    Applies the validated changes to the employee data.

    Args:
        emp_data: dict, the current employee data.
        validated_changes: dict, a dictionary of validated changes to apply.

    Returns:
        dict: An updated version of the employee dictionary.
    """
    updated_emp_data = emp_data.copy()
    for field, new_value in validated_changes.items():
        updated_emp_data[field] = new_value
    return updated_emp_data

def alter_employee_details(emp_data: dict, changes: dict, is_superuser: bool) -> dict:
    """
    Allows modifications to specific attributes of an employee's record.
    Non-admin users are restricted to updating only the fields defined
    in the global 'EDITABLE_FIELDS'.

    Args:
        emp_data: dict, a dictionary portraying the current state of the employee's details.
        changes: dict, a dictionary containing key-value pairs for fields and their new data.
        is_superuser: bool, a flag that indicates whether the person making the update is an admin or not.

    Returns:
        dict: An updated version of the employee dictionary, reflecting the valid changes.
    """
    user_type = "Superuser" if is_superuser else "User"
    logging.info(f"Attempting update by {user_type}.")
    logging.info(f"Original employee data: {json.dumps(emp_data, indent=2)}")
    logging.info(f"Requested changes: {json.dumps(changes, indent=2)}")

    validated_changes = _validate_changes(changes, is_superuser, EDITABLE_FIELDS)
    
    if not validated_changes:
        logging.warning(f"No valid changes to apply for user {user_type}. Original changes requested: {json.dumps(changes, indent=2)}")
        return emp_data # Return original data if no changes are valid

    updated_emp_data = _apply_changes(emp_data, validated_changes)

    if emp_data == updated_emp_data:
        logging.info(f"No actual changes made to employee data by {user_type} after validation. Validated changes were: {json.dumps(validated_changes, indent=2)}")
    else:
        logging.info(f"Changes applied by {user_type}. Validated changes: {json.dumps(validated_changes, indent=2)}")
        logging.info(f"Final employee data: {json.dumps(updated_emp_data, indent=2)}")
        
    return updated_emp_data
