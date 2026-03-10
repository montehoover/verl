ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def modify_car_details(car: dict, changes: dict) -> dict:
    for field, value in changes.items():
        if field in ALLOWED_FIELDS:
            car[field] = value
    return car
