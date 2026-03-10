ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def adjust_vehicle_info(auto, modifications):
    for field, value in modifications.items():
        if field in ALLOWED_FIELDS:
            auto[field] = value
    return auto
