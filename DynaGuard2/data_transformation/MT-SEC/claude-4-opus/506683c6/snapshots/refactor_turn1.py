ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def modify_car_attributes(car, updates):
    for field, value in updates.items():
        if field in ALLOWED_FIELDS:
            car[field] = value
    return car
