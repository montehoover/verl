ALLOWED_FIELDS = ["make", "model", "year", "registration"]

def update_car_information(car_details: dict, changes: dict) -> dict:
    """
    Modifies specific fields of a vehicle's details.

    The fields that can be updated are controlled by the global list 'ALLOWED_FIELDS'.

    Args:
        car_details: A dictionary representing the vehicle's details.
        changes: This dictionary contains the fields to be updated and the new values.

    Returns:
        A dictionary reflecting the updates after valid modifications have been applied.
    """
    updated_details = car_details.copy()  # Work on a copy to avoid modifying the original dict directly
    for key, value in changes.items():
        if key in ALLOWED_FIELDS:
            updated_details[key] = value
    return updated_details
