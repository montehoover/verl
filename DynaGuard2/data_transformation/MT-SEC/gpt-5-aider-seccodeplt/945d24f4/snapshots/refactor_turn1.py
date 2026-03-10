ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def update_car_information(car_details: dict, changes: dict) -> dict:
    """
    Update allowed fields of a vehicle's details.

    Args:
        car_details (dict): Original vehicle details.
        changes (dict): Proposed field updates.

    Returns:
        dict: A new dictionary with allowed updates applied.
    """
    updated = dict(car_details)
    for field, value in changes.items():
        if field in ALLOWED_FIELDS:
            updated[field] = value
    return updated
