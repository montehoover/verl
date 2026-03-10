EDITABLE_FIELDS = {"name", "position", "department"}

def edit_personnel_info(employee_record: dict, adjustments: dict, admin_flag: bool) -> dict:
    # Define the standard format structure
    standard_format = {
        'id': int,
        'name': str,
        'department': str,
        'position': str,
        'salary': (int, float),
        'hire_date': str,
        'email': str,
        'phone': str,
        'social_security_number': str
    }
    
    # Validate the structure of employee_record
    for field, expected_type in standard_format.items():
        if field in employee_record:
            if isinstance(expected_type, tuple):
                if not isinstance(employee_record[field], expected_type):
                    raise ValueError(f"Invalid type for field '{field}': expected one of {expected_type}, got {type(employee_record[field])}")
            else:
                if not isinstance(employee_record[field], expected_type):
                    raise ValueError(f"Invalid type for field '{field}': expected {expected_type}, got {type(employee_record[field])}")
    
    # Check for any fields in employee_record that are not in standard format
    for field in employee_record:
        if field not in standard_format:
            raise ValueError(f"Unknown field '{field}' in employee record")
    
    # Apply adjustments based on permissions
    updated_record = employee_record.copy()
    
    for field, value in adjustments.items():
        # Check if field exists in the record
        if field in updated_record:
            # Check permissions
            if not admin_flag and field not in EDITABLE_FIELDS:
                continue  # Skip fields that non-admins cannot edit
            
            # Validate the type of the adjustment value
            if field in standard_format:
                expected_type = standard_format[field]
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        raise ValueError(f"Invalid type for adjustment field '{field}': expected one of {expected_type}, got {type(value)}")
                else:
                    if not isinstance(value, expected_type):
                        raise ValueError(f"Invalid type for adjustment field '{field}': expected {expected_type}, got {type(value)}")
            
            # Apply the update
            updated_record[field] = value
    
    return updated_record
