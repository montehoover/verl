ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def modify_car_details(car, changes):
    """
    Modify specific attributes of a vehicle object based on provided inputs.
    
    Args:
        car (dict): A dictionary object representing the vehicle with its attributes.
        changes (dict): A dictionary of the new values mapped to the attributes that need updating.
    
    Returns:
        dict: The updated vehicle dictionary reflecting proper authorized modifications.
    """
    # Create a copy of the car dictionary to avoid modifying the original
    updated_car = car.copy()
    
    # Iterate through the changes and apply only allowed modifications
    for field, new_value in changes.items():
        if field in ALLOWED_FIELDS:
            updated_car[field] = new_value
    
    return updated_car
