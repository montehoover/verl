def display_vehicle_info(car, exclude_fields=None):
    """
    Display vehicle information in a formatted string.
    
    Args:
        car (dict): Dictionary containing vehicle attributes (make, model, year, owner, registration)
        exclude_fields (list, optional): List of field names to exclude from the display
    
    Returns:
        str: Formatted string with vehicle details
    """
    if exclude_fields is None:
        exclude_fields = []
    
    # Define the order and labels for display
    field_mapping = {
        'make': 'Make',
        'model': 'Model',
        'year': 'Year',
        'owner': 'Owner',
        'registration': 'Registration'
    }
    
    # Build the formatted string
    lines = ["Vehicle Information:"]
    lines.append("-" * 30)
    
    for field, label in field_mapping.items():
        if field not in exclude_fields and field in car:
            lines.append(f"{label}: {car[field]}")
    
    return "\n".join(lines)


# Example usage:
if __name__ == "__main__":
    # Test the function
    vehicle = {
        'make': 'Toyota',
        'model': 'Camry',
        'year': 2022,
        'owner': 'John Doe',
        'registration': 'ABC-1234'
    }
    
    # Display all information
    print(display_vehicle_info(vehicle))
    print("\n")
    
    # Display without owner information
    print(display_vehicle_info(vehicle, exclude_fields=['owner']))
