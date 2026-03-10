import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def display_vehicle_info(auto, fields_to_hide=None):
    """
    Display vehicle information in a formatted string.
    
    Args:
        auto (dict): Dictionary containing vehicle information with keys like
                     'make', 'model', 'year', 'owner', 'registration'
        fields_to_hide (list, optional): List of field names to exclude from display
    
    Returns:
        str: Formatted string containing vehicle details
    """
    if fields_to_hide is None:
        fields_to_hide = []
    
    # Define the display order and labels
    field_mapping = {
        'make': 'Make',
        'model': 'Model',
        'year': 'Year',
        'owner': 'Owner',
        'registration': 'Registration'
    }
    
    # Build the formatted output
    lines = ["Vehicle Information", "-" * 20]
    
    for field_key, field_label in field_mapping.items():
        if field_key not in fields_to_hide and field_key in auto:
            lines.append(f"{field_label}: {auto[field_key]}")
    
    return "\n".join(lines)


def log_and_update_vehicle(auto, modifications):
    """
    Update vehicle information and log all changes.
    
    Args:
        auto (dict): Dictionary containing current vehicle information
        modifications (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated vehicle dictionary with modifications applied
    """
    # Create a copy of the vehicle dictionary to avoid modifying the original
    updated_auto = auto.copy()
    
    # Process each modification
    for field, new_value in modifications.items():
        if field in updated_auto:
            old_value = updated_auto[field]
            # Only update and log if the value actually changes
            if old_value != new_value:
                updated_auto[field] = new_value
                logging.info(
                    f"Vehicle field '{field}' updated: '{old_value}' -> '{new_value}'"
                )
        else:
            # Add new field if it doesn't exist
            updated_auto[field] = new_value
            logging.info(
                f"Vehicle field '{field}' added with value: '{new_value}'"
            )
    
    return updated_auto
