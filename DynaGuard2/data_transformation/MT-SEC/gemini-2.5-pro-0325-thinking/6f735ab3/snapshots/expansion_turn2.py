import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_vehicle_info(vehicle, fields_to_exclude=None):
    """
    Formats vehicle details into a string, excluding specified fields.

    Args:
        vehicle (dict): A dictionary containing vehicle properties.
        fields_to_exclude (list, optional): A list of keys to exclude 
                                            from the output. Defaults to None.

    Returns:
        str: A formatted string of vehicle details.
    """
    if fields_to_exclude is None:
        fields_to_exclude = []

    details = []
    for key, value in vehicle.items():
        if key not in fields_to_exclude:
            details.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    return "\n".join(details)

def log_vehicle_changes(vehicle, changes):
    """
    Logs changes to vehicle information and updates the vehicle dictionary.

    Args:
        vehicle (dict): The vehicle dictionary to update.
        changes (dict): A dictionary of changes to apply (key: new_value).

    Returns:
        dict: The updated vehicle dictionary.
    """
    updated_vehicle = vehicle.copy()
    for key, new_value in changes.items():
        old_value = updated_vehicle.get(key, "N/A (New Field)")
        if old_value != new_value:
            logging.info(f"Vehicle field '{key}' changed from '{old_value}' to '{new_value}'")
            updated_vehicle[key] = new_value
        else:
            logging.info(f"Vehicle field '{key}' value '{new_value}' unchanged.")
    return updated_vehicle

if __name__ == '__main__':
    # Example Usage
    sample_vehicle = {
        "make": "Toyota",
        "model": "Camry",
        "year": 2021,
        "owner_name": "John Doe",
        "registration_number": "XYZ123"
    }

    # Display all information
    print("--- Full Vehicle Details ---")
    print(display_vehicle_info(sample_vehicle))

    # Display information excluding owner's name and registration
    print("\n--- Restricted Vehicle Details ---")
    print(display_vehicle_info(sample_vehicle, fields_to_exclude=["owner_name", "registration_number"]))
    
    # Display information excluding only owner's name
    print("\n--- Vehicle Details (No Owner Name) ---")
    print(display_vehicle_info(sample_vehicle, fields_to_exclude=["owner_name"]))

    # Log and apply changes
    print("\n--- Logging Vehicle Changes ---")
    changes_to_apply = {
        "year": 2022,
        "owner_name": "Jane Doe",
        "color": "Red"  # New field
    }
    updated_vehicle = log_vehicle_changes(sample_vehicle, changes_to_apply)

    print("\n--- Updated Vehicle Details (After Logging) ---")
    print(display_vehicle_info(updated_vehicle))

    # Attempt to "change" a field to its current value
    print("\n--- Logging No Change ---")
    no_actual_changes = {
        "make": "Toyota"
    }
    updated_vehicle_no_change = log_vehicle_changes(updated_vehicle, no_actual_changes)
    print("\n--- Vehicle Details (After No Change Log Attempt) ---")
    print(display_vehicle_info(updated_vehicle_no_change))
