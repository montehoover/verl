import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_vehicle_info(auto, fields_to_hide=None):
    """
    Displays vehicle information from a dictionary, with an option to hide certain fields.

    Args:
        auto (dict): A dictionary containing vehicle attributes.
                     Expected keys: 'make', 'model', 'year', 'owner', 'registration'.
        fields_to_hide (list, optional): A list of keys to exclude from the display.
                                         Defaults to None, meaning no fields are hidden.

    Returns:
        str: A formatted string containing the vehicle details.
    """
    if fields_to_hide is None:
        fields_to_hide = []

    display_items = []
    for key, value in auto.items():
        if key not in fields_to_hide:
            display_items.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    if not display_items:
        return "No vehicle information to display."
        
    return "\n".join(display_items)

def log_and_update_vehicle(auto, modifications):
    """
    Updates vehicle information based on a modifications dictionary and logs each change.

    Args:
        auto (dict): The vehicle dictionary to be updated.
        modifications (dict): A dictionary where keys are vehicle attributes to be updated
                              and values are the new values.

    Returns:
        dict: The updated vehicle dictionary.
    """
    logging.info(f"Starting update for vehicle: {auto.get('make', 'N/A')} {auto.get('model', 'N/A')}")
    for key, new_value in modifications.items():
        old_value = auto.get(key, 'N/A (new field)')
        if old_value != new_value:
            logging.info(f"Updating '{key}': from '{old_value}' to '{new_value}'")
            auto[key] = new_value
        else:
            logging.info(f"No change for '{key}': value is already '{new_value}'")
    logging.info(f"Finished update for vehicle: {auto.get('make', 'N/A')} {auto.get('model', 'N/A')}")
    return auto

if __name__ == '__main__':
    # Example Usage
    vehicle1 = {
        'make': 'Toyota',
        'model': 'Camry',
        'year': 2021,
        'owner': 'John Doe',
        'registration': 'ABC123XYZ'
    }

    vehicle2 = {
        'make': 'Honda',
        'model': 'Civic',
        'year': 2022,
        'owner': 'Jane Smith',
        'registration': 'DEF456UVW',
        'color': 'Blue'
    }

    print("--- Vehicle 1 (All Info) ---")
    print(display_vehicle_info(vehicle1))
    print("\n--- Vehicle 1 (Hide Owner) ---")
    print(display_vehicle_info(vehicle1, fields_to_hide=['owner']))
    print("\n--- Vehicle 1 (Hide Owner and Registration) ---")
    print(display_vehicle_info(vehicle1, fields_to_hide=['owner', 'registration']))

    print("\n--- Vehicle 2 (All Info with extra field) ---")
    print(display_vehicle_info(vehicle2))
    print("\n--- Vehicle 2 (Hide Registration and Color) ---")
    print(display_vehicle_info(vehicle2, fields_to_hide=['registration', 'color']))

    # Example with all sensitive fields hidden
    sensitive_fields = ['owner', 'registration']
    print(f"\n--- Vehicle 1 (Hide Sensitive: {', '.join(sensitive_fields)}) ---")
    print(display_vehicle_info(vehicle1, fields_to_hide=sensitive_fields))

    # Example with an empty vehicle dictionary
    empty_vehicle = {}
    print("\n--- Empty Vehicle ---")
    print(display_vehicle_info(empty_vehicle))

    # Example with all fields hidden
    all_fields = list(vehicle1.keys())
    print("\n--- Vehicle 1 (Hide All Fields) ---")
    print(display_vehicle_info(vehicle1, fields_to_hide=all_fields))

    print("\n--- Logging and Updating Vehicle 1 ---")
    vehicle1_modifications = {
        'year': 2022,
        'owner': 'Alice Wonderland',
        'color': 'Red' # Adding a new field
    }
    print(f"Original Vehicle 1: {vehicle1}")
    updated_vehicle1 = log_and_update_vehicle(vehicle1.copy(), vehicle1_modifications) # Use .copy() to preserve original for demo
    print(f"Updated Vehicle 1: {updated_vehicle1}")
    print("--- Displaying Updated Vehicle 1 (All Info) ---")
    print(display_vehicle_info(updated_vehicle1))

    print("\n--- Logging and Updating Vehicle 2 (No actual change for one field) ---")
    vehicle2_modifications = {
        'model': 'Civic', # No change
        'registration': 'GHI789JKL'
    }
    print(f"Original Vehicle 2: {vehicle2}")
    updated_vehicle2 = log_and_update_vehicle(vehicle2.copy(), vehicle2_modifications)
    print(f"Updated Vehicle 2: {updated_vehicle2}")
    print("--- Displaying Updated Vehicle 2 (Hide Owner) ---")
    print(display_vehicle_info(updated_vehicle2, fields_to_hide=['owner']))
