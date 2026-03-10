def edit_personnel_info(employee_record: dict, adjustments: dict) -> dict:
    """
    Validates and updates the structure of an employee's record.

    Args:
        employee_record: A dictionary representing the employee's current record.
        adjustments: A dictionary with updates to apply to the record.

    Returns:
        The updated employee record if the initial structure is valid.

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
    for key, value in adjustments.items():
        if key in updated_record:  # Only update if the key exists in the original record
            updated_record[key] = value
    
    return updated_record
