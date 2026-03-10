PERMITTED_FIELDS = ["price", "description", "stock"]


def is_permitted_field(field: str) -> bool:
    """
    Check if a field is permitted to be updated.

    Args:
        field (str): The field name to check.

    Returns:
        bool: True if the field is in PERMITTED_FIELDS, else False.
    """
    return field in PERMITTED_FIELDS


def extract_permitted_updates(update_info: dict) -> dict:
    """
    Extract only the permitted updates from the provided update_info.

    Args:
        update_info (dict): Incoming fields and values intended for update.

    Returns:
        dict: A dictionary containing only permitted fields and their values.
    """
    return {field: value for field, value in update_info.items() if is_permitted_field(field)}


def apply_updates(product_details: dict, permitted_updates: dict) -> dict:
    """
    Apply permitted updates to the product details without mutating the original.

    Args:
        product_details (dict): Original product information.
        permitted_updates (dict): Pre-filtered updates that are permitted.

    Returns:
        dict: A new product dictionary with updates applied.
    """
    updated = product_details.copy()
    updated.update(permitted_updates)
    return updated


def modify_product_data(product_details: dict, update_info: dict) -> dict:
    """
    Update the product details with fields from update_info that are permitted.

    Args:
        product_details (dict): Original product information.
        update_info (dict): Fields and values intended for update.

    Returns:
        dict: A new product dictionary with permitted changes applied.
    """
    if not isinstance(product_details, dict) or not isinstance(update_info, dict):
        raise TypeError("product_details and update_info must be dict instances")

    permitted = extract_permitted_updates(update_info)
    return apply_updates(product_details, permitted)
