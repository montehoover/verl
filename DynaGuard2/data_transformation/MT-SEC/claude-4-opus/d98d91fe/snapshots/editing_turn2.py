def modify_car_details(car, changes, user_role):
    """
    Modifies car details with type validation and access control.
    
    Args:
        car: dict - A dictionary with vehicle properties
        changes: dict - A dictionary with new attribute values
        user_role: str - The role of the user attempting to modify
        
    Returns:
        dict - The car dictionary with validated changes applied if authorized
    """
    # Check if user has permission to modify
    if user_role not in ['admin', 'editor']:
        return car
    
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
