import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALLOWED_FIELDS = ["make", "model", "year", "registration"]

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


def log_vehicle_changes(car, updates):
    """
    Logs changes to vehicle attributes and updates the vehicle dictionary.

    Args:
        car (dict): The vehicle dictionary to be updated.
        updates (dict): A dictionary containing the updates to apply.
                        Format: {'attribute_to_change': new_value}

    Returns:
        dict: The updated vehicle dictionary.
    """
    if not isinstance(car, dict):
        logging.error("Invalid 'car' object provided. Must be a dictionary.")
        return car # Or raise an error
    if not isinstance(updates, dict):
        logging.error("Invalid 'updates' object provided. Must be a dictionary.")
        return car # Or raise an error

    logging.info(f"Starting update process for vehicle: {car.get('make', 'N/A')} {car.get('model', 'N/A')}")
    for key, new_value in updates.items():
        old_value = car.get(key)
        if old_value != new_value:
            logging.info(f"Changing '{key}': from '{old_value}' to '{new_value}'")
            car[key] = new_value
        else:
            logging.info(f"No change for '{key}': value is already '{new_value}'")
    logging.info(f"Update process finished for vehicle: {car.get('make', 'N/A')} {car.get('model', 'N/A')}")
    return car

if __name__ == '__main__':
    # Example Usage for display_vehicle_info (existing)
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

    empty_vehicle = {}
    print("Empty Vehicle Information:")
    print(display_vehicle_info(empty_vehicle))
    print("\n" + "="*20 + "\n")

    print("Vehicle 1 Information (excluding make, model, year, owner, registration):")
    print(display_vehicle_info(vehicle1, exclude_fields=['make', 'model', 'year', 'owner', 'registration']))
    print("\n" + "="*40 + "\n")

    # Example Usage for log_vehicle_changes
    print("Logging changes for Vehicle 1:")
    vehicle1_updates = {
        'owner': 'Alice Wonderland',
        'year': 2022,
        'mileage': 15000 # New field
    }
    updated_vehicle1 = log_vehicle_changes(vehicle1.copy(), vehicle1_updates) # Use .copy() to preserve original vehicle1 for other tests
    print("\nUpdated Vehicle 1 Information:")
    print(display_vehicle_info(updated_vehicle1))
    print("\n" + "="*20 + "\n")

    print("Logging changes for Vehicle 2 (no actual change for year):")
    vehicle2_updates = {
        'color': 'Red',
        'year': 2020 # Same year
    }
    updated_vehicle2 = log_vehicle_changes(vehicle2.copy(), vehicle2_updates)
    print("\nUpdated Vehicle 2 Information:")
    print(display_vehicle_info(updated_vehicle2))
    print("\n" + "="*20 + "\n")

    # Example with non-dictionary inputs (error logging)
    print("Logging changes with invalid car input:")
    log_vehicle_changes("not_a_dict", {'owner': 'Test'})
    print("\n" + "="*20 + "\n")

    print("Logging changes with invalid updates input:")
    log_vehicle_changes(vehicle1.copy(), "not_a_dict")
    print("\n" + "="*40 + "\n")


def modify_car_attributes(car, updates):
    """
    Modifies vehicle attributes based on a list of allowed fields and logs changes.

    Args:
        car (dict): The vehicle dictionary to be updated.
        updates (dict): A dictionary containing the updates to apply.
                        Format: {'attribute_to_change': new_value}

    Returns:
        dict: The updated vehicle dictionary with only allowed fields modified.
    """
    if not isinstance(car, dict):
        logging.error("Invalid 'car' object provided to modify_car_attributes. Must be a dictionary.")
        return car # Or raise an error
    if not isinstance(updates, dict):
        logging.error("Invalid 'updates' object provided to modify_car_attributes. Must be a dictionary.")
        return car # Or raise an error

    valid_updates = {}
    for key, value in updates.items():
        if key in ALLOWED_FIELDS:
            valid_updates[key] = value
        else:
            logging.warning(f"Field '{key}' is not allowed for modification. Skipping.")

    if not valid_updates:
        logging.info("No valid fields to update in modify_car_attributes.")
        return car

    return log_vehicle_changes(car, valid_updates)

if __name__ == '__main__':
    # Example Usage for display_vehicle_info (existing)
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

    empty_vehicle = {}
    print("Empty Vehicle Information:")
    print(display_vehicle_info(empty_vehicle))
    print("\n" + "="*20 + "\n")

    print("Vehicle 1 Information (excluding make, model, year, owner, registration):")
    print(display_vehicle_info(vehicle1, exclude_fields=['make', 'model', 'year', 'owner', 'registration']))
    print("\n" + "="*40 + "\n")

    # Example Usage for log_vehicle_changes
    print("Logging changes for Vehicle 1:")
    vehicle1_updates_log = {
        'owner': 'Alice Wonderland',
        'year': 2022,
        'mileage': 15000 # New field
    }
    # Create a fresh copy for this specific test section
    vehicle1_for_log_test = vehicle1.copy()
    updated_vehicle1_log = log_vehicle_changes(vehicle1_for_log_test, vehicle1_updates_log)
    print("\nUpdated Vehicle 1 Information (after log_vehicle_changes):")
    print(display_vehicle_info(updated_vehicle1_log))
    print("\n" + "="*20 + "\n")

    print("Logging changes for Vehicle 2 (no actual change for year):")
    vehicle2_updates_log = {
        'color': 'Red',
        'year': 2020 # Same year
    }
    vehicle2_for_log_test = vehicle2.copy()
    updated_vehicle2_log = log_vehicle_changes(vehicle2_for_log_test, vehicle2_updates_log)
    print("\nUpdated Vehicle 2 Information (after log_vehicle_changes):")
    print(display_vehicle_info(updated_vehicle2_log))
    print("\n" + "="*20 + "\n")

    # Example with non-dictionary inputs (error logging for log_vehicle_changes)
    print("Logging changes with invalid car input (log_vehicle_changes):")
    log_vehicle_changes("not_a_dict", {'owner': 'Test'})
    print("\n" + "="*20 + "\n")

    print("Logging changes with invalid updates input (log_vehicle_changes):")
    log_vehicle_changes(vehicle1.copy(), "not_a_dict")
    print("\n" + "="*40 + "\n")

    # Example Usage for modify_car_attributes
    print("Modifying attributes for Vehicle 1 (year and owner - owner should be ignored):")
    # Create a fresh copy for this specific test section
    vehicle1_for_modify_test = vehicle1.copy()
    vehicle1_modify_updates = {
        'year': 2023,
        'owner': 'Bob The Builder', # This field is not in ALLOWED_FIELDS
        'registration': 'NEWREG123'
    }
    modified_vehicle1 = modify_car_attributes(vehicle1_for_modify_test, vehicle1_modify_updates)
    print("\nModified Vehicle 1 Information (after modify_car_attributes):")
    print(display_vehicle_info(modified_vehicle1))
    print("\n" + "="*20 + "\n")

    print("Modifying attributes for Vehicle 2 (only non-allowed field 'color'):")
    # Create a fresh copy for this specific test section
    vehicle2_for_modify_test = vehicle2.copy()
    vehicle2_modify_updates = {
        'color': 'Green', # This field is not in ALLOWED_FIELDS
        'engine_type': 'V8' # Also not allowed
    }
    modified_vehicle2 = modify_car_attributes(vehicle2_for_modify_test, vehicle2_modify_updates)
    print("\nModified Vehicle 2 Information (after modify_car_attributes - no changes expected):")
    print(display_vehicle_info(modified_vehicle2)) # Should be same as original vehicle2
    print(f"Original color: {vehicle2.get('color')}, Modified color: {modified_vehicle2.get('color')}")
    print("\n" + "="*20 + "\n")

    print("Modifying attributes with empty updates:")
    vehicle1_empty_updates = {}
    modified_vehicle1_empty = modify_car_attributes(vehicle1.copy(), vehicle1_empty_updates)
    print("\nModified Vehicle 1 with empty updates:")
    print(display_vehicle_info(modified_vehicle1_empty))
    print("\n" + "="*20 + "\n")

    print("Modifying attributes with invalid car input (modify_car_attributes):")
    modify_car_attributes("not_a_car_dict", {'year': 2024})
    print("\n" + "="*20 + "\n")

    print("Modifying attributes with invalid updates input (modify_car_attributes):")
    modify_car_attributes(vehicle1.copy(), "not_an_updates_dict")
