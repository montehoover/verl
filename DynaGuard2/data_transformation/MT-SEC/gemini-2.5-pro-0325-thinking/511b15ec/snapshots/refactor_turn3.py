import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PERMITTED_FIELDS = ["price", "description", "stock"]

def _validate_changes(change_data: dict, permitted_fields: list) -> dict:
    """
    Validates incoming data against a list of permitted fields.

    Args:
        change_data: dict, a dictionary containing the new values for the fields.
        permitted_fields: list, a list of fields that are allowed to be changed.

    Returns:
        A dictionary containing only the valid fields and their values from change_data.
    """
    if not isinstance(change_data, dict):
        logging.error("Invalid input: change_data is not a dictionary.")
        return {}
    if not isinstance(permitted_fields, list):
        logging.error("Invalid input: permitted_fields is not a list.")
        return {}

    validated_data = {}
    for field, value in change_data.items():
        if field not in permitted_fields:
            logging.warning(f"Field '{field}' is not permitted for update. Skipping.")
            continue
        validated_data[field] = value
    return validated_data

def _apply_updates(prod: dict, validated_data: dict) -> dict:
    """
    Applies validated changes to the product object.

    Args:
        prod: dict, the dictionary representing the product object.
        validated_data: dict, a dictionary containing validated field updates.

    Returns:
        A dictionary reflecting the changes made to the product object.
    """
    changes_made = {}
    for field, value in validated_data.items():
        if field not in prod or prod[field] != value:
            prod[field] = value
            changes_made[field] = value
    return changes_made

def update_product_info(prod: dict, change_data: dict) -> dict:
    """
    Updates certain fields of a product using incoming data.
    Only fields present in the PERMITTED_FIELDS list can be modified.

    Args:
        prod: dict, the dictionary representing the product object.
        change_data: dict, a dictionary containing the new values for the fields.

    Returns:
        A dictionary reflecting the changes made to the product object.
    """
    logging.info(f"Attempting to update product: {prod} with data: {change_data}")

    if not isinstance(prod, dict):
        logging.error("Invalid input: prod is not a dictionary.")
        return {}
    if not isinstance(change_data, dict):
        logging.error("Invalid input: change_data is not a dictionary.")
        return {}

    validated_data = _validate_changes(change_data, PERMITTED_FIELDS)
    logging.info(f"Validated change data: {validated_data}")

    if not validated_data:
        logging.info("No valid fields to update.")
        return {}

    changes_applied = _apply_updates(prod, validated_data)
    logging.info(f"Changes applied: {changes_applied}")
    logging.info(f"Product state after update: {prod}")
    return changes_applied
