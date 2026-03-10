def display_car_details(car_details, exclude_fields=None):
    """
    Formats and displays car details from a dictionary.

    Args:
        car_details (dict): A dictionary containing car properties.
                            Expected keys: 'make', 'model', 'year',
                                           'owner', 'registration'.
        exclude_fields (list, optional): A list of fields (keys) to exclude
                                         from the display. Defaults to None.

    Returns:
        str: A formatted string presenting the car details.
    """
    if exclude_fields is None:
        exclude_fields = []

    display_items = []
    for key, value in car_details.items():
        if key not in exclude_fields:
            # Capitalize the key and format the string
            display_items.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    return "\n".join(display_items)

if __name__ == '__main__':
    # Example Usage
    sample_car = {
        'make': 'Toyota',
        'model': 'Camry',
        'year': 2021,
        'owner': 'John Doe',
        'registration': 'XYZ123'
    }

    print("--- Displaying all details ---")
    print(display_car_details(sample_car))
    print("\n--- Displaying details excluding 'owner' ---")
    print(display_car_details(sample_car, exclude_fields=['owner']))
    print("\n--- Displaying details excluding 'owner' and 'registration' ---")
    print(display_car_details(sample_car, exclude_fields=['owner', 'registration']))
    print("\n--- Displaying details with an empty car dictionary ---")
    print(display_car_details({}))
    print("\n--- Displaying details with an empty car dictionary and exclusions ---")
    print(display_car_details({}, exclude_fields=['owner']))
