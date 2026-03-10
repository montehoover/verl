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
