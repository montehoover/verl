ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def alter_vehicle_info(details, modifications):
    # Create a copy of the details dictionary to avoid modifying the original
    updated_vehicle = details.copy()
    
    # Iterate through the modifications
    for field, value in modifications.items():
        # Only update if the field is in the allowed fields list
        if field in ALLOWED_FIELDS:
            updated_vehicle[field] = value
    
    return updated_vehicle
