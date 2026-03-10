PERMITTED_FIELDS = ["price", "description", "stock"]

def _validate_update_inputs(prod: dict, change_data: dict) -> None:
    """
    Pure validation function to ensure inputs are of the expected types.
    Raises TypeError on invalid inputs.
    """
    if not isinstance(prod, dict):
        raise TypeError("prod must be a dict")
    if not isinstance(change_data, dict):
        raise TypeError("change_data must be a dict")


def _compute_allowed_updates(change_data: dict, permitted_fields: list) -> dict:
    """
    Pure function that computes which updates are permitted without mutating inputs.
    Returns a new dict containing only permitted field updates.
    """
    return {key: value for key, value in change_data.items() if key in permitted_fields}


def update_product_info(prod: dict, change_data: dict) -> dict:
    """
    Update permitted fields of a product dict using values from change_data.

    Only keys listed in PERMITTED_FIELDS will be updated.
    Returns a dictionary of the fields that were updated with their new values.
    """
    _validate_update_inputs(prod, change_data)
    applied_changes = _compute_allowed_updates(change_data, PERMITTED_FIELDS)

    # Apply computed changes to the product (side-effect occurs here by design)
    for key, value in applied_changes.items():
        prod[key] = value

    return applied_changes
