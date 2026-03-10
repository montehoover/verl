import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_vehicle_info(details, fields_to_hide=None):
    """
    Display vehicle information from a dictionary.
    
    Args:
        details (dict): Vehicle details dictionary with keys like 'make', 'model', 'year', 'owner', 'registration'
        fields_to_hide (list): Optional list of field names to exclude from display
    
    Returns:
        str: Formatted string containing vehicle information
    """
    if fields_to_hide is None:
        fields_to_hide = []
    
    # Define display labels for better formatting
    field_labels = {
        'make': 'Make',
        'model': 'Model',
        'year': 'Year',
        'owner': 'Owner',
        'registration': 'Registration'
    }
    
    # Build the formatted string
    lines = ["Vehicle Information:"]
    lines.append("-" * 20)
    
    for field, label in field_labels.items():
        if field in details and field not in fields_to_hide:
            lines.append(f"{label}: {details[field]}")
    
    return "\n".join(lines)


def log_vehicle_changes(details, modifications):
    """
    Update vehicle details and log all changes made.
    
    Args:
        details (dict): Current vehicle details dictionary
        modifications (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated vehicle dictionary with all modifications applied
    """
    # Create a copy to avoid modifying the original
    updated_details = details.copy()
    
    # Log header for this update session
    if modifications:
        logging.info(f"Vehicle changes initiated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if 'registration' in details:
            logging.info(f"Vehicle Registration: {details['registration']}")
    
    # Process each modification
    for field, new_value in modifications.items():
        old_value = details.get(field, "Not set")
        
        # Update the field
        updated_details[field] = new_value
        
        # Log the change
        logging.info(f"Field '{field}' changed from '{old_value}' to '{new_value}'")
    
    # Log completion
    if modifications:
        logging.info("Vehicle update completed")
    
    return updated_details
