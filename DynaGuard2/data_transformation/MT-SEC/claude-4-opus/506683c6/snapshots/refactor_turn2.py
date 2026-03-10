ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def is_field_allowed(field, allowed_fields):
    """Check if a field is in the list of allowed fields."""
    return field in allowed_fields

def update_car_field(car, field, value):
    """Update a single field in the car dictionary."""
    car[field] = value
    return car

def filter_allowed_updates(updates, allowed_fields):
    """Filter updates to only include allowed fields."""
    return {field: value for field, value in updates.items() if is_field_allowed(field, allowed_fields)}

def modify_car_attributes(car, updates):
    allowed_updates = filter_allowed_updates(updates, ALLOWED_FIELDS)
    for field, value in allowed_updates.items():
        car = update_car_field(car, field, value)
    return car
