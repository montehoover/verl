def view_vehicle_details(car: dict, fields_to_exclude: list = None) -> str:
    """
    Formats vehicle details for display, excluding specified fields.

    Args:
        car: A dictionary representing a vehicle with attributes like
             make, model, year, owner, and registration.
        fields_to_exclude: A list of keys (strings) to exclude from the
                           output. Defaults to ['registration'].

    Returns:
        A formatted string presenting the vehicle details.
    """
    if fields_to_exclude is None:
        fields_to_exclude = ['registration']

    details = []
    for key, value in car.items():
        if key not in fields_to_exclude:
            details.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    return "\n".join(details)

if __name__ == '__main__':
    # Example Usage
    sample_car = {
        "make": "Toyota",
        "model": "Camry",
        "year": 2021,
        "owner": "John Doe",
        "registration": "XYZ123",
        "color": "Blue"
    }

    print("--- Displaying all details (excluding default 'registration') ---")
    print(view_vehicle_details(sample_car))
    print("\n--- Displaying all details (explicitly excluding 'registration' and 'owner') ---")
    print(view_vehicle_details(sample_car, fields_to_exclude=['registration', 'owner']))
    print("\n--- Displaying all details (explicitly excluding nothing) ---")
    print(view_vehicle_details(sample_car, fields_to_exclude=[]))
