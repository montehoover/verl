import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define editable fields for non-admin users
EDITABLE_FIELDS = {"name", "position", "department"}

def display_employee_info(staff, include_sensitive=False):
    """
    Display employee information in a formatted string.
    
    Args:
        staff (dict): Employee dictionary with attributes like name, position, 
                     salary, department, and social_security_number
        include_sensitive (bool): Whether to include sensitive information 
                                 like social_security_number (default: False)
    
    Returns:
        str: Formatted string with employee details
    """
    output = []
    output.append(f"Employee: {staff.get('name', 'N/A')}")
    output.append(f"Position: {staff.get('position', 'N/A')}")
    output.append(f"Department: {staff.get('department', 'N/A')}")
    output.append(f"Salary: ${staff.get('salary', 0):,.2f}")
    
    if include_sensitive and 'social_security_number' in staff:
        output.append(f"SSN: {staff['social_security_number']}")
    
    return "\n".join(output)


def update_and_log_employee(staff, changes):
    """
    Update employee record with changes and log each modification.
    
    Args:
        staff (dict): Employee dictionary to be updated
        changes (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated employee dictionary
    """
    # Create a copy to avoid modifying the original
    updated_staff = staff.copy()
    
    # Get employee name for logging
    employee_name = staff.get('name', 'Unknown Employee')
    
    # Update each field and log the change
    for field, new_value in changes.items():
        old_value = updated_staff.get(field, 'Not set')
        updated_staff[field] = new_value
        
        # Log the change
        logging.info(f"Updated {employee_name}'s {field}: '{old_value}' → '{new_value}'")
    
    # Log summary of changes
    if changes:
        logging.info(f"Employee record updated for {employee_name} - {len(changes)} field(s) modified")
    else:
        logging.info(f"No changes made to {employee_name}'s record")
    
    return updated_staff


def modify_staff_info(staff, changes, admin_status):
    """
    Modify employee information with role-based restrictions.
    
    Args:
        staff (dict): Employee dictionary to be modified
        changes (dict): Dictionary containing fields to update with new values
        admin_status (bool): True if user is admin, False otherwise
    
    Returns:
        dict: Updated employee dictionary
    """
    # Create a copy to avoid modifying the original
    updated_staff = staff.copy()
    
    # Get employee name for logging
    employee_name = staff.get('name', 'Unknown Employee')
    
    # Filter changes based on admin status
    if admin_status:
        # Admin can modify any field
        allowed_changes = changes
    else:
        # Non-admin can only modify fields in EDITABLE_FIELDS
        allowed_changes = {field: value for field, value in changes.items() if field in EDITABLE_FIELDS}
        
        # Log any restricted fields
        restricted_fields = set(changes.keys()) - EDITABLE_FIELDS
        if restricted_fields:
            for field in restricted_fields:
                logging.warning(f"Access denied: Non-admin user cannot modify '{field}' for {employee_name}")
    
    # Apply allowed changes
    for field, new_value in allowed_changes.items():
        old_value = updated_staff.get(field, 'Not set')
        updated_staff[field] = new_value
        
        # Log the change
        logging.info(f"Modified {employee_name}'s {field}: '{old_value}' → '{new_value}' (Admin: {admin_status})")
    
    # Log summary
    if allowed_changes:
        logging.info(f"Employee record modified for {employee_name} - {len(allowed_changes)} field(s) updated")
    else:
        logging.info(f"No modifications made to {employee_name}'s record")
    
    return updated_staff
