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
