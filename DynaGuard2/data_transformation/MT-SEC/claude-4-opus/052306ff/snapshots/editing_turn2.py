from datetime import datetime

def log_change(field_name, old_value, new_value):
    print(f"Field {field_name} changed from {old_value} to {new_value} at {datetime.now().isoformat()}")

def alter_employee_details(emp_data: dict, changes: dict, is_superuser: bool) -> tuple[bool, dict]:
    """
    Check employee data for consistency and apply changes if user is authorized.
    
    Args:
        emp_data: Dictionary containing employee information
        changes: Dictionary containing potential updates to employee data
        is_superuser: Boolean indicating if user has admin privileges
        
    Returns:
        tuple: (bool indicating if data is valid, updated employee dict)
    """
    # Create a copy of employee data to potentially update
    updated_emp_data = emp_data.copy()
    
    # Validate existing data
    if 'name' in emp_data and not isinstance(emp_data['name'], str):
        return False, emp_data
    
    if 'position' in emp_data and not isinstance(emp_data['position'], str):
        return False, emp_data
    
    if 'salary' in emp_data and not isinstance(emp_data['salary'], (int, float)):
        return False, emp_data
    
    # Validate changes
    for field, new_value in changes.items():
        if field == 'name' and not isinstance(new_value, str):
            return False, emp_data
        
        if field == 'position' and not isinstance(new_value, str):
            return False, emp_data
        
        if field == 'salary' and not isinstance(new_value, (int, float)):
            return False, emp_data
    
    # Apply changes if user is superuser
    if is_superuser:
        for field, new_value in changes.items():
            if field in updated_emp_data:
                old_value = updated_emp_data[field]
                if old_value != new_value:
                    log_change(field, old_value, new_value)
                    updated_emp_data[field] = new_value
            else:
                log_change(field, None, new_value)
                updated_emp_data[field] = new_value
    
    return True, updated_emp_data
