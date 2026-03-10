import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def display_vehicle_info(details, fields_to_hide=None):
    """
    Formats and displays vehicle information from a dictionary.

    Args:
        details (dict): A dictionary containing vehicle attributes.
                        Expected keys: 'make', 'model', 'year', 
                                       'owner', 'registration'.
        fields_to_hide (list or set, optional): A list or set of field names 
                                                (keys from the details dict) 
                                                to exclude from the output. 
                                                Defaults to None (no fields hidden).

    Returns:
        str: A formatted string presenting the vehicle's details.
    """
    if fields_to_hide is None:
        fields_to_hide = set()
    else:
        fields_to_hide = set(fields_to_hide)

    display_items = []
    # Define a preferred order for display, can be extended
    preferred_order = ['make', 'model', 'year', 'registration', 'owner']

    # Add items in preferred order if they are not hidden
    for key in preferred_order:
        if key in details and key not in fields_to_hide:
            display_items.append(f"{key.replace('_', ' ').capitalize()}: {details[key]}")

    # Add any remaining items not in preferred order and not hidden
    for key, value in details.items():
        if key not in preferred_order and key not in fields_to_hide:
            display_items.append(f"{key.replace('_', ' ').capitalize()}: {value}")
            
    if not display_items:
        return "No vehicle information to display."

    return "\n".join(display_items)

if __name__ == '__main__':
    # Example Usage
    vehicle1_details = {
        'make': 'Toyota',
        'model': 'Camry',
        'year': 2021,
        'owner': 'John Doe',
        'registration': 'ABC123XYZ'
    }

    vehicle2_details = {
        'make': 'Honda',
        'model': 'Civic',
        'year': 2020,
        'owner': 'Jane Smith',
        'registration': 'DEF456UVW',
        'color': 'Blue'
    }

    print("Vehicle 1 Information (Full Details):")
    print(display_vehicle_info(vehicle1_details))
    print("\n" + "="*30 + "\n")

    print("Vehicle 1 Information (Hiding Owner):")
    print(display_vehicle_info(vehicle1_details, fields_to_hide=['owner']))
    print("\n" + "="*30 + "\n")

    print("Vehicle 2 Information (Full Details with extra field):")
    print(display_vehicle_info(vehicle2_details))
    print("\n" + "="*30 + "\n")

    print("Vehicle 2 Information (Hiding Owner and Registration):")
    print(display_vehicle_info(vehicle2_details, fields_to_hide={'owner', 'registration'}))
    print("\n" + "="*30 + "\n")
    
    print("Vehicle Information (Hiding all known fields):")
    print(display_vehicle_info(vehicle1_details, fields_to_hide=['make', 'model', 'year', 'owner', 'registration']))
    print("\n" + "="*30 + "\n")

    empty_vehicle = {}
    print("Empty Vehicle Information:")
    print(display_vehicle_info(empty_vehicle))
    print("\n" + "="*30 + "\n")

    empty_vehicle_hide_owner = {}
    print("Empty Vehicle Information (Hiding Owner):")
    print(display_vehicle_info(empty_vehicle_hide_owner, fields_to_hide=['owner']))
    print("\n" + "="*30 + "\n")


def log_vehicle_changes(details, modifications):
    """
    Updates vehicle details with modifications and logs each change.

    Args:
        details (dict): The current vehicle information dictionary.
        modifications (dict): A dictionary of changes to apply. 
                              Keys are field names, values are new field values.

    Returns:
        dict: The updated vehicle information dictionary.
    """
    updated_details = details.copy() # Work on a copy to avoid modifying the original dict directly if passed by reference elsewhere
    
    for key, new_value in modifications.items():
        old_value = updated_details.get(key, 'N/A (New Field)')
        if old_value != new_value:
            logging.info(f"Vehicle field '{key}' changed from '{old_value}' to '{new_value}'.")
            updated_details[key] = new_value
        else:
            logging.info(f"Vehicle field '{key}' attempted update with same value '{new_value}'. No change made.")
            
    return updated_details

if __name__ == '__main__':
    # Example Usage for display_vehicle_info (existing)
    vehicle1_details = {
        'make': 'Toyota',
        'model': 'Camry',
        'year': 2021,
        'owner': 'John Doe',
        'registration': 'ABC123XYZ'
    }

    vehicle2_details = {
        'make': 'Honda',
        'model': 'Civic',
        'year': 2020,
        'owner': 'Jane Smith',
        'registration': 'DEF456UVW',
        'color': 'Blue'
    }

    print("Vehicle 1 Information (Full Details):")
    print(display_vehicle_info(vehicle1_details))
    print("\n" + "="*30 + "\n")

    print("Vehicle 1 Information (Hiding Owner):")
    print(display_vehicle_info(vehicle1_details, fields_to_hide=['owner']))
    print("\n" + "="*30 + "\n")

    print("Vehicle 2 Information (Full Details with extra field):")
    print(display_vehicle_info(vehicle2_details))
    print("\n" + "="*30 + "\n")

    print("Vehicle 2 Information (Hiding Owner and Registration):")
    print(display_vehicle_info(vehicle2_details, fields_to_hide={'owner', 'registration'}))
    print("\n" + "="*30 + "\n")
    
    print("Vehicle Information (Hiding all known fields):")
    print(display_vehicle_info(vehicle1_details, fields_to_hide=['make', 'model', 'year', 'owner', 'registration']))
    print("\n" + "="*30 + "\n")

    empty_vehicle = {}
    print("Empty Vehicle Information:")
    print(display_vehicle_info(empty_vehicle))
    print("\n" + "="*30 + "\n")

    empty_vehicle_hide_owner = {}
    print("Empty Vehicle Information (Hiding Owner):")
    print(display_vehicle_info(empty_vehicle_hide_owner, fields_to_hide=['owner']))
    print("\n" + "="*30 + "\n")

    # Example Usage for log_vehicle_changes
    print("Logging Vehicle Changes Example:")
    vehicle_to_modify = {
        'make': 'Ford',
        'model': 'Ranger',
        'year': 2019,
        'owner': 'Alice Brown',
        'registration': 'GHI789JKL'
    }
    print("Original Vehicle Details:")
    print(display_vehicle_info(vehicle_to_modify))
    
    modifications1 = {
        'year': 2020,
        'owner': 'Alice Green'
    }
    print(f"\nApplying modifications: {modifications1}")
    vehicle_to_modify = log_vehicle_changes(vehicle_to_modify, modifications1)
    print("Updated Vehicle Details:")
    print(display_vehicle_info(vehicle_to_modify))
    print("\n" + "="*30 + "\n")

    modifications2 = {
        'registration': 'NEWREG123',
        'color': 'Red', # Adding a new field
        'owner': 'Alice Green' # Attempting to update with the same value
    }
    print(f"\nApplying modifications: {modifications2}")
    vehicle_to_modify = log_vehicle_changes(vehicle_to_modify, modifications2)
    print("Further Updated Vehicle Details:")
    print(display_vehicle_info(vehicle_to_modify))
