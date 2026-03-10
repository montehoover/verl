import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
