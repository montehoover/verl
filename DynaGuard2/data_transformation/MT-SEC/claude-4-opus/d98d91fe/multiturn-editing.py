ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def modify_car_details(car, changes):
    """
    Modifies car details with validation against allowed fields.
    
    Args:
        car: dict - A dictionary with vehicle properties
        changes: dict - A dictionary with new attribute values
        
    Returns:
        dict - The car dictionary with authorized changes applied
    """
    for key, value in changes.items():
        if key in ALLOWED_FIELDS:
            if key in ['make', 'model']:
                if isinstance(value, str):
                    car[key] = value
            elif key == 'year':
                if isinstance(value, int):
                    car[key] = value
            elif key == 'registration':
                if isinstance(value, str):
                    car[key] = value
    
    return car
