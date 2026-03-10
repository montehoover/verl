ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def is_field_allowed(field):
    """Check if a field is in the allowed fields list."""
    return field in ALLOWED_FIELDS

def apply_modifications(vehicle, modifications):
    """Apply valid modifications to a vehicle dictionary."""
    updated_vehicle = vehicle.copy()
    
    for field, value in modifications.items():
        if is_field_allowed(field):
            updated_vehicle[field] = value
    
    return updated_vehicle

def alter_vehicle_info(details, modifications):
    """Update specific fields of a vehicle's attributes using given input data."""
    return apply_modifications(details, modifications)
