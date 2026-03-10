ALLOWED_FIELDS = ["make", "model", "year", "registration"]

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


def update_car_information(car_details, changes):
    """
    Updates car information based on allowed fields.

    Args:
        car_details (dict): The original car details dictionary.
        changes (dict): A dictionary of proposed changes. Only changes to
                        fields listed in ALLOWED_FIELDS will be applied.

    Returns:
        dict: The updated car details dictionary.
    """
    updated_details = car_details.copy()
    print("--- Updating Car Information (Allowed Fields Only) ---")
    for key, value in changes.items():
        if key in ALLOWED_FIELDS:
            old_value = updated_details.get(key, 'N/A (New Field)')
            updated_details[key] = value
            print(f"Updating '{key}': From '{old_value}' to '{value}'")
        else:
            print(f"Skipping update for '{key}': Not an allowed field.")
    print("--- End of Update ---")
    return updated_details

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

    print("\n--- Updating car information with allowed fields ---")
    # Create a fresh car dictionary for this example to avoid side effects from previous updates
    car_for_selective_update = {
        'make': 'Honda',
        'model': 'Civic',
        'year': 2020,
        'owner': 'Alice Wonderland',
        'registration': 'ABC789',
        'color': 'Blue'
    }
    print("Original car details for selective update:")
    print(display_car_details(car_for_selective_update))
    
    changes_to_apply = {
        'model': 'Accord',      # Allowed
        'year': 2021,           # Allowed
        'owner': 'Bob The Builder', # Not allowed
        'color': 'Green',       # Not allowed
        'registration': 'DEF456' # Allowed
    }
    
    selectively_updated_car = update_car_information(car_for_selective_update, changes_to_apply)
    print("\n--- Displaying selectively updated car details ---")
    print(display_car_details(selectively_updated_car))
