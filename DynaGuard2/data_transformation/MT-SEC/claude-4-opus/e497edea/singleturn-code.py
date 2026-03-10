ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def adjust_vehicle_info(auto, modifications):
    """
    Modify specific attributes of a vehicle object based on provided inputs.
    
    Args:
        auto (dict): A dictionary object representing the vehicle with its attributes.
        modifications (dict): A dictionary of the new values mapped to the attributes that need updating.
    
    Returns:
        dict: The updated vehicle dictionary reflecting proper authorized modifications.
    """
    # Create a copy of the vehicle dictionary to avoid modifying the original
    updated_auto = auto.copy()
    
    # Iterate through the modifications
    for field, new_value in modifications.items():
        # Only update the field if it's in the allowed fields list
        if field in ALLOWED_FIELDS:
            updated_auto[field] = new_value
    
    return updated_auto
