from typing import Tuple

def edit_personnel_info(employee_record: dict, adjustments: dict) -> Tuple[dict, int]:
    """
    Validates and updates the structure of an employee's record and counts changed fields.

    Args:
        employee_record: A dictionary representing the employee's current record.
        adjustments: A dictionary with updates to apply to the record.

    Returns:
        A tuple containing the updated employee record and the number of fields changed.

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
    changed_fields_count = 0
    for key, value in adjustments.items():
        if key in updated_record:  # Only update if the key exists in the original record
            if updated_record[key] != value: # Check if the value is actually changing
                updated_record[key] = value
                changed_fields_count += 1
    
    return updated_record, changed_fields_count
