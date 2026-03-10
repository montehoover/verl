def edit_personnel_info(employee_record: dict, adjustments: dict) -> tuple[dict, int]:
    # Define the standard format structure
    standard_format = {
        'id': int,
        'name': str,
        'department': str,
        'position': str,
        'salary': (int, float),
        'hire_date': str,
        'email': str,
        'phone': str
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
    
    # Apply adjustments only to existing fields and count changes
    updated_record = employee_record.copy()
    modified_count = 0
    
    for field, value in adjustments.items():
        if field in updated_record:
            # Validate the type of the adjustment value
            if field in standard_format:
                expected_type = standard_format[field]
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        raise ValueError(f"Invalid type for adjustment field '{field}': expected one of {expected_type}, got {type(value)}")
                else:
                    if not isinstance(value, expected_type):
                        raise ValueError(f"Invalid type for adjustment field '{field}': expected {expected_type}, got {type(value)}")
            
            # Check if the value actually changed
            if updated_record[field] != value:
                updated_record[field] = value
                modified_count += 1
    
    return updated_record, modified_count
