ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def is_allowed_field(field: str) -> bool:
    return field in ALLOWED_FIELDS

def update_car_attribute(car: dict, field: str, value) -> dict:
    if is_allowed_field(field):
        car[field] = value
    return car

def filter_allowed_changes(changes: dict) -> dict:
    return {field: value for field, value in changes.items() if is_allowed_field(field)}

def modify_car_details(car: dict, changes: dict) -> dict:
    allowed_changes = filter_allowed_changes(changes)
    
    if not allowed_changes:
        return car
    
    for field, value in allowed_changes.items():
        car = update_car_attribute(car, field, value)
    
    return car
