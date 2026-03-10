ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def update_car_information(car_details, changes):
    updated_car = car_details.copy()
    
    for field, value in changes.items():
        if field in ALLOWED_FIELDS:
            updated_car[field] = value
    
    return updated_car
