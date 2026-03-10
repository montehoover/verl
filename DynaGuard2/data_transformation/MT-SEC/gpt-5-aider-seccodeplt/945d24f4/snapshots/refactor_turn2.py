ALLOWED_FIELDS = ["make", "model", "year", "registration"]


def is_allowed_field(field: str) -> bool:
    """
    Determine if a field is allowed to be updated.

    Args:
        field (str): The field name to check.

    Returns:
        bool: True if the field is allowed, False otherwise.
    """
    return field in ALLOWED_FIELDS


def filter_allowed_changes(changes: dict) -> dict:
    """
    Filter the provided changes to include only allowed fields.

    Args:
        changes (dict): Proposed field updates.

    Returns:
        dict: A new dictionary containing only allowed updates.
    """
    return {field: value for field, value in changes.items() if is_allowed_field(field)}


def apply_updates(car_details: dict, allowed_changes: dict) -> dict:
    """
    Apply allowed changes to the car details without mutating inputs.

    Args:
        car_details (dict): Original vehicle details.
        allowed_changes (dict): Validated changes that are allowed.

    Returns:
        dict: A new dictionary with the allowed changes applied.
    """
    updated = dict(car_details)
    updated.update(allowed_changes)
    return updated


def update_car_information(car_details: dict, changes: dict) -> dict:
    """
    Update allowed fields of a vehicle's details.

    Args:
        car_details (dict): Original vehicle details.
        changes (dict): Proposed field updates.

    Returns:
        dict: A new dictionary with allowed updates applied.
    """
    allowed_changes = filter_allowed_changes(changes)
    return apply_updates(car_details, allowed_changes)
