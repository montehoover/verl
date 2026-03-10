ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def update_car_information(car_details, changes):
    # Create a copy of the car_details to avoid modifying the original
    updated_car = car_details.copy()
    
    # Iterate through the changes dictionary
    for field, new_value in changes.items():
        # Only update if the field is in ALLOWED_FIELDS
        if field in ALLOWED_FIELDS:
            updated_car[field] = new_value
    
    return updated_car
