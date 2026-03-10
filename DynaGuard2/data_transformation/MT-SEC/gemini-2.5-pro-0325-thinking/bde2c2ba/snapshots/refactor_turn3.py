import logging

PERMITTED_FIELDS = ["price", "description", "stock"]

def _filter_permitted_data(new_data: dict, allowed_fields: list) -> dict:
    """
    Filters new_data to include only keys present in allowed_fields.
    """
    return {key: value for key, value in new_data.items() if key in allowed_fields}

def _apply_updates(product_info: dict, data_to_update: dict) -> dict:
    """
    Applies updates from data_to_update to a copy of product_info.
    """
    updated_product_info = product_info.copy()
    updated_product_info.update(data_to_update)
    return updated_product_info

def update_item_information(product_info: dict, new_data: dict) -> dict:
    """
    Modifies product attributes based on information from a provided data dictionary.
    Updates are permitted only for the fields present in the global list PERMITTED_FIELDS.
    Logs the updates being made.

    Args:
        product_info: A dictionary instance representing a product.
        new_data: A dictionary containing key-value pairs representing intended
                  updates to the product details.

    Returns:
        A dictionary object reflecting the updated product state.
    """
    # Configure basic logging if no handlers are configured for the root logger
    # This makes the logging work out-of-the-box if not configured elsewhere.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)

    permitted_updates = _filter_permitted_data(new_data, PERMITTED_FIELDS)

    if permitted_updates:
        logger.info(f"Updating product information. Changes: {permitted_updates}")
    else:
        logger.info("No valid fields to update or no new data provided for update.")

    updated_product_info = _apply_updates(product_info, permitted_updates)
    return updated_product_info
