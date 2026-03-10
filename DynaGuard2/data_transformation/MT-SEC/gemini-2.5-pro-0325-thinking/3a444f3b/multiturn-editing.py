from typing import Tuple

EDITABLE_FIELDS = {"name", "position", "department"}

def edit_personnel_info(employee_record: dict, adjustments: dict, admin_flag: bool) -> dict:
    """
    Validates and updates the structure of an employee's record based on admin privileges.

    Args:
        employee_record: A dictionary representing the employee's current record.
        adjustments: A dictionary with updates to apply to the record.
        admin_flag: A boolean indicating if the user has admin privileges.

    Returns:
        The updated employee record reflecting permissible changes.

    Raises:
        ValueError: If the employee_record does not conform to the standard structure.
    """
    STANDARD_KEYS = {'name', 'employee_id', 'department'}

    # Validate structure: Check if all standard keys are present in the employee_record
    if not STANDARD_KEYS.issubset(employee_record.keys()):
        missing_keys = STANDARD_KEYS - employee_record.keys()
        raise ValueError(f"Employee record is missing required keys: {missing_keys}")

    # Apply adjustments for existing fields only
    # Work on a copy to avoid modifying the original dict if it's passed by reference
    # and used elsewhere, and to adhere to "fields not present in the initial record should not be added"
    updated_record = employee_record.copy()
    # changed_fields_count = 0 # No longer counting changed fields as per new requirements
    for key, value in adjustments.items():
        if key in updated_record:  # Only update if the key exists in the original record
            can_edit_field = admin_flag or key in EDITABLE_FIELDS
            if can_edit_field:
                # if updated_record[key] != value: # Check if the value is actually changing - this check is not strictly needed for the update logic itself
                updated_record[key] = value
                # changed_fields_count += 1 # No longer counting
    
    return updated_record
