ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def is_field_allowed(field):
    return field in ALLOWED_FIELDS

def apply_modification(auto, field, value):
    auto[field] = value
    return auto

def filter_allowed_modifications(modifications):
    return {field: value for field, value in modifications.items() if is_field_allowed(field)}

def adjust_vehicle_info(auto, modifications):
    allowed_modifications = filter_allowed_modifications(modifications)
    for field, value in allowed_modifications.items():
        apply_modification(auto, field, value)
    return auto
