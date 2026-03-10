import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def display_vehicle_info(vehicle, fields_to_exclude=None):
    """
    Display vehicle information in a formatted string.
    
    Args:
        vehicle (dict): Dictionary containing vehicle information with keys like
                       'make', 'model', 'year', 'owner', 'registration'
        fields_to_exclude (list, optional): List of field names to exclude from display
    
    Returns:
        str: Formatted string with vehicle details
    """
    if fields_to_exclude is None:
        fields_to_exclude = []
    
    # Define display labels for each field
    field_labels = {
        'make': 'Make',
        'model': 'Model',
        'year': 'Year',
        'owner': 'Owner',
        'registration': 'Registration'
    }
    
    # Build the formatted output
    lines = ["Vehicle Information:"]
    lines.append("-" * 30)
    
    for field, label in field_labels.items():
        if field in vehicle and field not in fields_to_exclude:
            value = vehicle[field]
            lines.append(f"{label}: {value}")
    
    return "\n".join(lines)


def log_vehicle_changes(vehicle, changes):
    """
    Log changes made to vehicle information and update the vehicle dictionary.
    
    Args:
        vehicle (dict): Original vehicle dictionary
        changes (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated vehicle dictionary
    """
    # Create a copy of the vehicle to avoid modifying the original
    updated_vehicle = vehicle.copy()
    
    # Log header for this change session
    logging.info(f"Vehicle changes initiated for Registration: {vehicle.get('registration', 'N/A')}")
    
    # Process each change
    for field, new_value in changes.items():
        if field in updated_vehicle:
            old_value = updated_vehicle[field]
            
            # Only log and update if the value actually changed
            if old_value != new_value:
                logging.info(f"Field '{field}' changed from '{old_value}' to '{new_value}'")
                updated_vehicle[field] = new_value
            else:
                logging.info(f"Field '{field}' unchanged (value: '{old_value}')")
        else:
            # New field being added
            logging.info(f"New field '{field}' added with value '{new_value}'")
            updated_vehicle[field] = new_value
    
    logging.info("Vehicle changes completed")
    
    return updated_vehicle


def update_vehicle_info(vehicle, updates):
    """
    Update vehicle attributes based on specific permissions.
    
    Args:
        vehicle (dict): Original vehicle dictionary
        updates (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated vehicle dictionary with only allowed field modifications
    """
    # Create a copy of the vehicle to avoid modifying the original
    updated_vehicle = vehicle.copy()
    
    # Filter updates to only include allowed fields
    allowed_updates = {}
    for field, value in updates.items():
        if field in ALLOWED_FIELDS:
            allowed_updates[field] = value
        else:
            logging.warning(f"Attempted to update restricted field '{field}' - update ignored")
    
    # Apply allowed updates
    for field, new_value in allowed_updates.items():
        if field in updated_vehicle:
            old_value = updated_vehicle[field]
            updated_vehicle[field] = new_value
            logging.info(f"Updated '{field}' from '{old_value}' to '{new_value}'")
        else:
            updated_vehicle[field] = new_value
            logging.info(f"Added new field '{field}' with value '{new_value}'")
    
    return updated_vehicle
