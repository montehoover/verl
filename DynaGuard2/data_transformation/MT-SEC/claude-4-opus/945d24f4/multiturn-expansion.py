ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def display_car_details(car_details, exclude_fields=None):
    """
    Display car details in a formatted string.
    
    Args:
        car_details (dict): Dictionary containing car information
        exclude_fields (list): Optional list of fields to exclude from display
    
    Returns:
        str: Formatted string of car details
    """
    if exclude_fields is None:
        exclude_fields = []
    
    # Define the order and labels for display
    field_labels = {
        'make': 'Make',
        'model': 'Model',
        'year': 'Year',
        'owner': 'Owner',
        'registration': 'Registration'
    }
    
    # Build the formatted string
    lines = []
    lines.append("=== Vehicle Details ===")
    
    for field, label in field_labels.items():
        if field not in exclude_fields and field in car_details:
            lines.append(f"{label}: {car_details[field]}")
    
    lines.append("=====================")
    
    return '\n'.join(lines)


def log_and_update_car(car_details, changes):
    """
    Update car details and log all changes made.
    
    Args:
        car_details (dict): Original car details dictionary
        changes (dict): Dictionary of changes to apply
    
    Returns:
        dict: Updated car details dictionary
    """
    # Create a copy to avoid modifying the original
    updated_car = car_details.copy()
    
    print("=== Change Log ===")
    
    for field, new_value in changes.items():
        if field in updated_car:
            old_value = updated_car[field]
            if old_value != new_value:
                print(f"Updated {field}: {old_value} -> {new_value}")
                updated_car[field] = new_value
            else:
                print(f"No change for {field}: value remains {old_value}")
        else:
            print(f"Added new field {field}: {new_value}")
            updated_car[field] = new_value
    
    print("=================")
    
    return updated_car


def update_car_information(car_details, changes):
    """
    Update car information based on allowed fields only.
    
    Args:
        car_details (dict): Original car details dictionary
        changes (dict): Dictionary of changes to apply
    
    Returns:
        dict: Updated car details dictionary with only allowed fields changed
    """
    # Create a copy to avoid modifying the original
    updated_car = car_details.copy()
    
    for field, new_value in changes.items():
        if field in ALLOWED_FIELDS:
            updated_car[field] = new_value
    
    return updated_car


# Example usage
if __name__ == "__main__":
    car = {
        'make': 'Toyota',
        'model': 'Camry',
        'year': 2021,
        'owner': 'John Doe',
        'registration': 'ABC-123'
    }
    
    # Display all details
    print(display_car_details(car))
    print()
    
    # Display without owner information
    print(display_car_details(car, exclude_fields=['owner']))
    print()
    
    # Update car details
    changes = {
        'owner': 'Jane Smith',
        'year': 2022,
        'color': 'Blue'
    }
    
    updated_car = log_and_update_car(car, changes)
    print()
    print("Updated car details:")
    print(display_car_details(updated_car))
    
    # Test update_car_information with restricted fields
    print("\n=== Testing update_car_information ===")
    restricted_changes = {
        'make': 'Honda',
        'model': 'Accord',
        'owner': 'Alice Johnson',  # This should be ignored
        'year': 2023,
        'color': 'Red'  # This should be ignored
    }
    
    restricted_updated_car = update_car_information(car, restricted_changes)
    print("After restricted update:")
    print(display_car_details(restricted_updated_car))
