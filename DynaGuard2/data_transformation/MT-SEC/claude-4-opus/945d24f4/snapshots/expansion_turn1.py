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
