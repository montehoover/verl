ALLOWED_FIELDS = ["make", "model", "year", "registration"]

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

def log_vehicle_changes(car: dict, changes: dict) -> dict:
    """
    Updates vehicle details and logs changes.

    Args:
        car: The vehicle dictionary to update.
        changes: A dictionary where keys are field names to change
                 and values are the new values.

    Returns:
        The updated vehicle dictionary.
    """
    print("\n--- Logging Vehicle Changes ---")
    for key, new_value in changes.items():
        original_value = car.get(key, "N/A (New Field)")
        print(f"Changing '{key}': From '{original_value}' to '{new_value}'")
        car[key] = new_value
    print("--- End of Changes ---")
    return car

def modify_car_details(car: dict, changes: dict) -> dict:
    """
    Modifies vehicle details for allowed fields only.

    Args:
        car: The vehicle dictionary to update.
        changes: A dictionary where keys are field names to change
                 and values are the new values.

    Returns:
        The updated vehicle dictionary.
    """
    print("\n--- Modifying Car Details (Allowed Fields Only) ---")
    for key, new_value in changes.items():
        if key in ALLOWED_FIELDS:
            original_value = car.get(key, "N/A (New Field)")
            print(f"Updating '{key}': From '{original_value}' to '{new_value}'")
            car[key] = new_value
        else:
            print(f"Skipping '{key}': Field not allowed for modification.")
    print("--- End of Modifications ---")
    return car

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

    # Example Usage for log_vehicle_changes
    updated_car = log_vehicle_changes(sample_car, {"owner": "Jane Doe", "year": 2022, "mileage": 15000})
    print("\n--- Displaying details after changes ---")
    print(view_vehicle_details(updated_car))

    # Example with adding a new field and changing an existing one
    further_updated_car = log_vehicle_changes(updated_car, {"color": "Red", "service_due_date": "2025-12-31"})
    print("\n--- Displaying details after further changes ---")
    print(view_vehicle_details(further_updated_car))

    # Example Usage for modify_car_details
    car_to_modify = {
        "make": "Honda",
        "model": "Civic",
        "year": 2020,
        "owner": "Alice Wonderland",
        "registration": "ABC789",
        "color": "Black"
    }
    print("\n--- Original Car for Modification ---")
    print(view_vehicle_details(car_to_modify, fields_to_exclude=[]))
    
    modified_car = modify_car_details(car_to_modify, {"year": 2021, "owner": "Bob The Builder", "registration": "NEWREG123"})
    print("\n--- Displaying details after modification (allowed fields) ---")
    print(view_vehicle_details(modified_car, fields_to_exclude=[]))

    # Attempt to modify a non-allowed field along with an allowed one
    another_modified_car = modify_car_details(modified_car, {"make": "Honder (Typo Corrected)", "color": "Silver"})
    print("\n--- Displaying details after attempting to modify non-allowed field ---")
    print(view_vehicle_details(another_modified_car, fields_to_exclude=[]))
