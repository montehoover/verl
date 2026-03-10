ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def update_vehicle_info(vehicle, updates):
    updated_vehicle = vehicle.copy()
    for field, value in updates.items():
        if field in ALLOWED_FIELDS:
            updated_vehicle[field] = value
    return updated_vehicle
