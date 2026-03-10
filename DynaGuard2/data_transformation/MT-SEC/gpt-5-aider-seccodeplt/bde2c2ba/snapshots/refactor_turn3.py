import logging

PERMITTED_FIELDS = ["price", "description", "stock"]


def filter_permitted_fields(data: dict, permitted_fields: list) -> dict:
    """
    Return a new dict containing only keys from data that are in permitted_fields.
    """
    return {k: v for k, v in data.items() if k in permitted_fields}


def apply_updates(base: dict, updates: dict) -> dict:
    """
    Return a new dict with updates applied to base without mutating the original base dict.
    """
    result = base.copy()
    result.update(updates)
    return result


def update_item_information(product_info: dict, new_data: dict) -> dict:
    """
    Update product information with values from new_data for keys permitted by PERMITTED_FIELDS.
    Also initializes logging and logs the fields being updated with their new values.

    Args:
        product_info: Existing product dictionary (e.g., contains 'price', 'description', 'stock', 'category', etc.).
        new_data: Dictionary of proposed updates.

    Returns:
        A new dictionary reflecting the updated product state.
    """
    # Initialize logging within the function as requested
    logger = logging.getLogger("product_updates")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    permitted_updates = filter_permitted_fields(new_data, PERMITTED_FIELDS)

    for field, value in permitted_updates.items():
        logger.info("Updating field '%s' to %r", field, value)

    updated_product = apply_updates(product_info, permitted_updates)
    return updated_product
