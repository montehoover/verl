def modify_car_details(car, changes):
    """
    Modifies car details with type validation.
    
    Args:
        car: dict - A dictionary with vehicle properties
        changes: dict - A dictionary with new attribute values
        
    Returns:
        dict - The car dictionary with validated changes applied
    """
    for key, value in changes.items():
        if key in ['make', 'model']:
            if isinstance(value, str):
                car[key] = value
        elif key == 'year':
            if isinstance(value, int):
                car[key] = value
        else:
            # For other attributes, apply without type checking
            car[key] = value
    
    return car
