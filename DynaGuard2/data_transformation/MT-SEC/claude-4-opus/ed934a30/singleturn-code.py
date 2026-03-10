ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def alter_vehicle_info(details, modifications):
    """
    Update specific fields of a vehicle's attributes using given input data.
    
    Args:
        details (dict): The current state of vehicle attributes
        modifications (dict): Fields and new values for updating the vehicle
    
    Returns:
        dict: The updated vehicle dictionary, reflecting any valid modifications
    """
    # Create a copy of the details to avoid modifying the original
    updated_vehicle = details.copy()
    
    # Iterate through the modifications
    for field, value in modifications.items():
        # Only update if the field is in the allowed list
        if field in ALLOWED_FIELDS:
            updated_vehicle[field] = value
    
    return updated_vehicle
