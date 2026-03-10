from datetime import datetime

EDITABLE_FIELDS = {"name", "position", "department"}

def log_change(field_name, old_value, new_value):
    print(f"Field {field_name} changed from {old_value} to {new_value} at {datetime.now().isoformat()}")

def alter_employee_details(emp_data: dict, changes: dict, is_superuser: bool) -> tuple[bool, dict]:
    """
    Update employee details based on user permissions.
    
    Args:
        emp_data: Dictionary containing employee information
        changes: Dictionary containing potential updates to employee data
        is_superuser: Boolean indicating if user has admin privileges
        
    Returns:
        tuple: (bool indicating if updates were successful, updated employee dict)
    """
    # Create a copy of employee data to update
    updated_emp_data = emp_data.copy()
    
    # Apply changes based on user permissions
    for field, new_value in changes.items():
        if is_superuser:
            # Admin users can update any field
            updated_emp_data[field] = new_value
        elif field in EDITABLE_FIELDS:
            # Non-admin users can only update fields in EDITABLE_FIELDS
            updated_emp_data[field] = new_value
    
    return True, updated_emp_data
