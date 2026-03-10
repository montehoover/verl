def display_vehicle_info(car, exclude_fields=None):
    """
    Displays vehicle details from a dictionary.

    Args:
        car (dict): A dictionary containing vehicle attributes like
                    'make', 'model', 'year', 'owner', and 'registration'.
        exclude_fields (list, optional): A list of fields (keys) to exclude
                                         from the display. Defaults to None.

    Returns:
        str: A formatted string presenting the vehicle details.
    """
    if exclude_fields is None:
        exclude_fields = []

    details = []
    # Define a preferred order for display, can be extended
    ordered_keys = ['make', 'model', 'year', 'registration', 'owner']

    for key in ordered_keys:
        if key in car and key not in exclude_fields:
            # Capitalize the key and replace underscores for better readability
            formatted_key = key.replace('_', ' ').capitalize()
            details.append(f"{formatted_key}: {car[key]}")

    # Add any other keys that might be in the car dict but not in ordered_keys
    for key, value in car.items():
        if key not in ordered_keys and key not in exclude_fields:
            formatted_key = key.replace('_', ' ').capitalize()
            details.append(f"{formatted_key}: {value}")

    if not details:
        return "No vehicle information to display."

    return "\n".join(details)

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
        'year': 2020,
        'owner': 'Jane Smith',
        'registration': 'DEF456UVW',
        'color': 'Blue' # Additional field
    }

    print("Vehicle 1 Information:")
    print(display_vehicle_info(vehicle1))
    print("\n" + "="*20 + "\n")

    print("Vehicle 1 Information (excluding owner):")
    print(display_vehicle_info(vehicle1, exclude_fields=['owner']))
    print("\n" + "="*20 + "\n")

    print("Vehicle 2 Information:")
    print(display_vehicle_info(vehicle2))
    print("\n" + "="*20 + "\n")

    print("Vehicle 2 Information (excluding owner and registration):")
    print(display_vehicle_info(vehicle2, exclude_fields=['owner', 'registration']))
    print("\n" + "="*20 + "\n")

    # Example with an empty car dictionary
    empty_vehicle = {}
    print("Empty Vehicle Information:")
    print(display_vehicle_info(empty_vehicle))
    print("\n" + "="*20 + "\n")

    # Example excluding all known fields
    print("Vehicle 1 Information (excluding make, model, year, owner, registration):")
    print(display_vehicle_info(vehicle1, exclude_fields=['make', 'model', 'year', 'owner', 'registration']))
