ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def filter_allowed_updates(updates, allowed_fields):
    return {field: value for field, value in updates.items() if field in allowed_fields}

def apply_updates(vehicle, filtered_updates):
    updated_vehicle = vehicle.copy()
    updated_vehicle.update(filtered_updates)
    return updated_vehicle

def update_vehicle_info(vehicle, updates):
    filtered_updates = filter_allowed_updates(updates, ALLOWED_FIELDS)
    return apply_updates(vehicle, filtered_updates)
