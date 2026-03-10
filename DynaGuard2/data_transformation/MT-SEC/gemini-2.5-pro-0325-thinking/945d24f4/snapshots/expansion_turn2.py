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


def log_and_update_car(car_details, changes):
    """
    Updates car details based on a changes dictionary and logs each change.

    Args:
        car_details (dict): The original car details dictionary.
        changes (dict): A dictionary where keys are car attributes to be updated
                        and values are the new values.

    Returns:
        dict: The updated car details dictionary.
    """
    print("--- Logging Changes ---")
    for key, new_value in changes.items():
        old_value = car_details.get(key, 'N/A (New Field)')
        print(f"Updating '{key}': From '{old_value}' to '{new_value}'")
        car_details[key] = new_value
    print("--- End of Changes ---")
    return car_details

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

    print("\n--- Modifying car details ---")
    updated_car = log_and_update_car(sample_car, {'year': 2022, 'owner': 'Jane Doe', 'color': 'Red'})
    print("\n--- Displaying updated car details ---")
    print(display_car_details(updated_car))

    print("\n--- Modifying car details again (e.g., registration change) ---")
    further_updated_car = log_and_update_car(updated_car, {'registration': 'NEWREG1', 'mileage': 15000})
    print("\n--- Displaying further updated car details ---")
    print(display_car_details(further_updated_car))
