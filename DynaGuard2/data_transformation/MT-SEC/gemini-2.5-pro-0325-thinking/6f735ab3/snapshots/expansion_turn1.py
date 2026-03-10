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
