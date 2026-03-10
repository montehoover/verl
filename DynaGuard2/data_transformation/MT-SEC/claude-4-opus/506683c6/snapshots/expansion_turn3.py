import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define allowed fields for modification
ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def display_vehicle_info(car, exclude_fields=None):
    """
    Display vehicle information in a formatted string.
    
    Args:
        car (dict): Dictionary containing vehicle attributes (make, model, year, owner, registration)
        exclude_fields (list, optional): List of field names to exclude from the display
    
    Returns:
        str: Formatted string with vehicle details
    """
    if exclude_fields is None:
        exclude_fields = []
    
    # Define the order and labels for display
    field_mapping = {
        'make': 'Make',
        'model': 'Model',
        'year': 'Year',
        'owner': 'Owner',
        'registration': 'Registration'
    }
    
    # Build the formatted string
    lines = ["Vehicle Information:"]
    lines.append("-" * 30)
    
    for field, label in field_mapping.items():
        if field not in exclude_fields and field in car:
            lines.append(f"{label}: {car[field]}")
    
    return "\n".join(lines)


def log_vehicle_changes(car, updates):
    """
    Log changes made to vehicle attributes and update the vehicle dictionary.
    
    Args:
        car (dict): Original vehicle dictionary
        updates (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated vehicle dictionary
    """
    # Create a copy of the vehicle dictionary to avoid modifying the original
    updated_car = car.copy()
    
    # Track changes
    changes_made = []
    
    for field, new_value in updates.items():
        if field in updated_car:
            old_value = updated_car[field]
            if old_value != new_value:
                # Log the change
                logging.info(f"Vehicle {updated_car.get('registration', 'Unknown')}: {field} changed from '{old_value}' to '{new_value}'")
                changes_made.append({
                    'field': field,
                    'old_value': old_value,
                    'new_value': new_value,
                    'timestamp': datetime.now().isoformat()
                })
                # Update the value
                updated_car[field] = new_value
        else:
            # New field being added
            logging.info(f"Vehicle {updated_car.get('registration', 'Unknown')}: New field '{field}' added with value '{new_value}'")
            changes_made.append({
                'field': field,
                'old_value': None,
                'new_value': new_value,
                'timestamp': datetime.now().isoformat()
            })
            updated_car[field] = new_value
    
    # Store change history in the vehicle dictionary
    if 'change_history' not in updated_car:
        updated_car['change_history'] = []
    updated_car['change_history'].extend(changes_made)
    
    return updated_car


def modify_car_attributes(car, updates):
    """
    Modify vehicle attributes while adhering to specific constraints.
    Only allows modifications to fields specified in ALLOWED_FIELDS.
    
    Args:
        car (dict): Original vehicle dictionary
        updates (dict): Dictionary containing fields to update with new values
    
    Returns:
        dict: Updated vehicle dictionary with only valid changes applied
    """
    # Create a copy of the vehicle dictionary to avoid modifying the original
    updated_car = car.copy()
    
    # Filter updates to only include allowed fields
    valid_updates = {}
    
    for field, new_value in updates.items():
        if field in ALLOWED_FIELDS:
            valid_updates[field] = new_value
            logging.info(f"Modification allowed for field '{field}'")
        else:
            logging.warning(f"Modification denied for field '{field}' - not in ALLOWED_FIELDS")
    
    # Apply valid updates using the existing log_vehicle_changes function
    if valid_updates:
        updated_car = log_vehicle_changes(updated_car, valid_updates)
    
    return updated_car


# Example usage:
if __name__ == "__main__":
    # Test the function
    vehicle = {
        'make': 'Toyota',
        'model': 'Camry',
        'year': 2022,
        'owner': 'John Doe',
        'registration': 'ABC-1234'
    }
    
    # Display all information
    print(display_vehicle_info(vehicle))
    print("\n")
    
    # Display without owner information
    print(display_vehicle_info(vehicle, exclude_fields=['owner']))
    print("\n")
    
    # Test vehicle changes
    updates = {
        'owner': 'Jane Smith',
        'year': 2023,
        'color': 'Blue'
    }
    
    print("Logging vehicle changes...")
    updated_vehicle = log_vehicle_changes(vehicle, updates)
    print("\n")
    
    # Display updated vehicle info
    print(display_vehicle_info(updated_vehicle))
    print("\n")
    
    # Test modify_car_attributes with both allowed and disallowed fields
    print("Testing modify_car_attributes...")
    restricted_updates = {
        'make': 'Honda',
        'model': 'Accord',
        'year': 2024,
        'owner': 'Alice Johnson',  # This should be denied
        'color': 'Red'  # This should be denied
    }
    
    modified_vehicle = modify_car_attributes(vehicle, restricted_updates)
    print("\n")
    
    # Display modified vehicle info
    print(display_vehicle_info(modified_vehicle))
