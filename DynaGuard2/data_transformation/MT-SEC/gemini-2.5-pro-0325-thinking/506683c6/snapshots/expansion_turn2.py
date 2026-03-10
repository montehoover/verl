import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
