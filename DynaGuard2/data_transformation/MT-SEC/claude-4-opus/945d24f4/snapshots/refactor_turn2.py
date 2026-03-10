ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def is_field_allowed(field, allowed_fields):
    return field in allowed_fields

def apply_field_updates(car_details, changes, allowed_fields):
    updated_car = car_details.copy()
    
    for field, value in changes.items():
        if is_field_allowed(field, allowed_fields):
            updated_car[field] = value
    
    return updated_car

def update_car_information(car_details, changes):
    return apply_field_updates(car_details, changes, ALLOWED_FIELDS)
