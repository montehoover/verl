PERMITTED_FIELDS = ["price", "description", "stock"]

def update_product_info(prod: dict, change_data: dict) -> dict:
    """
    Update permitted fields of a product dict using values from change_data.

    Only keys listed in PERMITTED_FIELDS will be updated.
    Returns a dictionary of the fields that were updated with their new values.
    """
    if not isinstance(prod, dict):
        raise TypeError("prod must be a dict")
    if not isinstance(change_data, dict):
        raise TypeError("change_data must be a dict")

    applied_changes = {}

    for key, value in change_data.items():
        if key in PERMITTED_FIELDS:
            prod[key] = value
            applied_changes[key] = value

    return applied_changes
