import logging
import json

PERMITTED_FIELDS = ["price", "description", "stock"]

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _to_json(data) -> str:
    """Serialize data to a JSON string for clear logging."""
    try:
        return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        # Fallback to repr if JSON serialization fails
        return repr(data)


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

    logger.info("Starting product update")
    logger.info("Original product details: %s", _to_json(product_details))
    logger.info("Update information received: %s", _to_json(update_info))

    permitted = extract_permitted_updates(update_info)
    updated = apply_updates(product_details, permitted)

    logger.info("Updated product details: %s", _to_json(updated))
    logger.debug("Permitted updates applied: %s", _to_json(permitted))
    logger.debug("Fields updated: %s", _to_json(list(permitted.keys())))

    return updated
