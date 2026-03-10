from typing import Dict, Any, Tuple
from datetime import datetime

def log_change(field_name, old_value, new_value):
    print(f"Field {field_name} changed from {old_value} to {new_value} at {datetime.now().isoformat()}")

def alter_employee_details(emp_data: Dict[str, Any], changes: Dict[str, Any], is_superuser: bool) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates changes, logs them, and updates employee data if the user is a superuser.

    Args:
        emp_data: The current employee data dictionary.
        changes: A dictionary containing proposed changes (field_name: new_value).
        is_superuser: True if the user has administrative privileges to apply changes.

    Returns:
        A tuple (all_changes_were_type_valid, updated_emp_data):
        - all_changes_were_type_valid (bool): True if all processed changes to 'name', 
                                              'position', or 'salary' had valid types, 
                                              False otherwise.
        - updated_emp_data (Dict[str, Any]): The employee data dictionary, updated with
                                             valid changes if is_superuser was True.
    """
    updated_emp_data = emp_data.copy()
    all_changes_were_type_valid = True

    for field_name, new_value in changes.items():
        # Determine if the field is one we validate and process ('name', 'position', 'salary')
        is_known_field_for_processing = field_name in ("name", "position", "salary")
        
        if not is_known_field_for_processing:
            # Ignore fields not explicitly handled for validation, logging, or update
            continue

        current_change_type_valid = False
        if field_name == "name":
            if isinstance(new_value, str):
                current_change_type_valid = True
        elif field_name == "position":
            if isinstance(new_value, str):
                current_change_type_valid = True
        elif field_name == "salary":
            if isinstance(new_value, (int, float)):
                current_change_type_valid = True

        if current_change_type_valid:
            old_value = updated_emp_data.get(field_name) # Get old value from current state of updated_emp_data
            log_change(field_name, old_value, new_value)
            if is_superuser:
                updated_emp_data[field_name] = new_value
        else:
            # A known field ('name', 'position', 'salary') had an invalid type
            all_changes_were_type_valid = False
            # Do not log or apply this invalid change for a known field

    return all_changes_were_type_valid, updated_emp_data
