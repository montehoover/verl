ALLOWED_FIELDS = ['make', 'model', 'year', 'registration']

def restricted_update(original: dict, updates: dict, allowed_fields: list[str]) -> dict:
    """
    Update 'original' with key/value pairs from 'updates' for keys present in 'allowed_fields'.

    Only keys listed in 'allowed_fields' will be updated or added; all other keys in 'updates' are ignored.
    This function mutates 'original' in place and returns it.
    """
    allowed = set(allowed_fields)
    for key, value in updates.items():
        if key in allowed:
            original[key] = value
    return original

def adjust_vehicle_info(auto: dict, modifications: dict) -> dict:
    """
    Update 'auto' with key/value pairs from 'modifications' only for fields allowed by ALLOWED_FIELDS.

    This function mutates 'auto' in place and returns it.
    """
    return restricted_update(auto, modifications, ALLOWED_FIELDS)
