import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

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
    print("\n" + "="*30 + "\n")


def alter_vehicle_info(details, modifications):
    """
    Updates vehicle details with modifications, but only for allowed fields.
    Logs valid changes.

    Args:
        details (dict): The current vehicle information dictionary.
        modifications (dict): A dictionary of changes to apply.
                              Keys are field names, values are new field values.

    Returns:
        dict: The updated vehicle information dictionary with allowed changes.
    """
    updated_details = details.copy()
    
    for key, new_value in modifications.items():
        if key in ALLOWED_FIELDS:
            old_value = updated_details.get(key, 'N/A (New Field)')
            if old_value != new_value:
                logging.info(f"Vehicle field '{key}' (allowed) changed from '{old_value}' to '{new_value}'.")
                updated_details[key] = new_value
            else:
                logging.info(f"Vehicle field '{key}' (allowed) attempted update with same value '{new_value}'. No change made.")
        else:
            logging.warning(f"Attempt to modify unauthorized field '{key}'. Change not applied.")
            
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

    # Example Usage for log_vehicle_changes (existing)
    print("Logging Vehicle Changes Example:")
    vehicle_to_modify_log = { # Renamed to avoid conflict with alter_vehicle_info example
        'make': 'Ford',
        'model': 'Ranger',
        'year': 2019,
        'owner': 'Alice Brown',
        'registration': 'GHI789JKL'
    }
    print("Original Vehicle Details (for log_vehicle_changes):")
    print(display_vehicle_info(vehicle_to_modify_log))
    
    modifications1_log = {
        'year': 2020,
        'owner': 'Alice Green'
    }
    print(f"\nApplying modifications (log_vehicle_changes): {modifications1_log}")
    vehicle_to_modify_log = log_vehicle_changes(vehicle_to_modify_log, modifications1_log)
    print("Updated Vehicle Details (log_vehicle_changes):")
    print(display_vehicle_info(vehicle_to_modify_log))
    print("\n" + "="*30 + "\n")

    modifications2_log = {
        'registration': 'NEWREG123',
        'color': 'Red', 
        'owner': 'Alice Green'
    }
    print(f"\nApplying modifications (log_vehicle_changes): {modifications2_log}")
    vehicle_to_modify_log = log_vehicle_changes(vehicle_to_modify_log, modifications2_log)
    print("Further Updated Vehicle Details (log_vehicle_changes):")
    print(display_vehicle_info(vehicle_to_modify_log))
    print("\n" + "="*30 + "\n")

    # Example Usage for alter_vehicle_info
    print("Altering Vehicle Info Example (with constraints):")
    vehicle_to_alter = {
        'make': 'Nissan',
        'model': 'Altima',
        'year': 2018,
        'owner': 'Bob White',
        'registration': 'MNO123PQR',
        'color': 'Silver'
    }
    print("Original Vehicle Details (for alter_vehicle_info):")
    print(display_vehicle_info(vehicle_to_alter))

    alterations1 = {
        'year': 2019,               # Allowed
        'owner': 'Bob Black',       # Not allowed
        'registration': 'STU456VWX' # Allowed
    }
    print(f"\nApplying alterations: {alterations1}")
    vehicle_to_alter = alter_vehicle_info(vehicle_to_alter, alterations1)
    print("Updated Vehicle Details (alter_vehicle_info):")
    print(display_vehicle_info(vehicle_to_alter))
    print("\n" + "="*30 + "\n")

    alterations2 = {
        'model': 'Maxima',          # Allowed
        'color': 'Black',           # Not allowed
        'mileage': 50000            # Not allowed (new field)
    }
    print(f"\nApplying alterations: {alterations2}")
    vehicle_to_alter = alter_vehicle_info(vehicle_to_alter, alterations2)
    print("Further Updated Vehicle Details (alter_vehicle_info):")
    print(display_vehicle_info(vehicle_to_alter))
