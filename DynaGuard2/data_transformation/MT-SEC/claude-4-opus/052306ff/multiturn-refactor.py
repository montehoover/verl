import json
import datetime

EDITABLE_FIELDS = {"name", "position", "department"}

def validate_field_access(field, is_superuser):
    """Check if a field can be modified based on user permissions."""
    return is_superuser or field in EDITABLE_FIELDS

def filter_valid_changes(changes, is_superuser):
    """Filter changes to only include fields the user has permission to modify."""
    return {
        field: value 
        for field, value in changes.items() 
        if validate_field_access(field, is_superuser)
    }

def apply_changes(emp_data, valid_changes):
    """Apply validated changes to employee data."""
    updated_employee = emp_data.copy()
    updated_employee.update(valid_changes)
    return updated_employee

def log_employee_update(emp_data, changes, updated_employee, is_superuser):
    """Log employee update details to a file in the current directory."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_type": "superuser" if is_superuser else "regular",
        "employee_id": emp_data.get("id", "unknown"),
        "changes_requested": changes,
        "changes_applied": {k: v for k, v in changes.items() if k in updated_employee and updated_employee[k] == v},
        "final_state": updated_employee
    }
    
    with open("employee_updates.log", "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

def alter_employee_details(emp_data, changes, is_superuser):
    valid_changes = filter_valid_changes(changes, is_superuser)
    updated_employee = apply_changes(emp_data, valid_changes)
    log_employee_update(emp_data, changes, updated_employee, is_superuser)
    return updated_employee
