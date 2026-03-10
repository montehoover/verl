ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def update_vehicle_info(vehicle, updates):
    """
    Update vehicle attributes based on input data.
    
    Args:
        vehicle: dict, a dictionary representing the vehicle object.
        updates: dict, a dictionary containing the attributes to be updated and their new values.
    
    Returns:
        A dictionary representing the updated vehicle object.
    """
    # Create a copy of the vehicle to avoid modifying the original
    updated_vehicle = vehicle.copy()
    
    # Iterate through the updates and apply only allowed fields
    for field, value in updates.items():
        if field in ALLOWED_FIELDS:
            updated_vehicle[field] = value
    
    return updated_vehicle
