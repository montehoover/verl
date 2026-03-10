import logging

PERMITTED_FIELDS = ["price", "description", "stock"]
_PERMITTED_FIELDS_SET = frozenset(PERMITTED_FIELDS)


def is_permitted_field(field: str) -> bool:
    """
    Check if a field is allowed to be updated.

    Pure function: does not mutate inputs or rely on external state changes.
    """
    return field in _PERMITTED_FIELDS_SET


def extract_permitted_updates(details: dict) -> dict:
    """
    Return a new dict containing only permitted fields from the provided details.

    Pure function: does not mutate inputs.
    """
    return {field: value for field, value in details.items() if is_permitted_field(field)}


def merge_item_with_updates(item: dict, updates: dict) -> dict:
    """
    Return a new product dict with updates applied.

    Pure function: does not mutate inputs.
    """
    merged = item.copy()
    merged.update(updates)
    return merged


def edit_product_info(item: dict, details: dict) -> dict:
    """
    Update permitted fields of a product dictionary based on provided details.

    Args:
        item (dict): The original product dictionary.
        details (dict): A dictionary containing fields and values intended for update.

    Returns:
        dict: A new product dictionary with permitted changes applied.
    """
    if not isinstance(item, dict):
        raise TypeError("item must be a dict")
    if not isinstance(details, dict):
        raise TypeError("details must be a dict")

    # Initialize logging (human-readable) within this function
    logger = logging.getLogger("product_edit")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    permitted_updates = extract_permitted_updates(details)

    # Determine which fields will actually change
    changed_fields = {k: v for k, v in permitted_updates.items() if item.get(k) != v}

    if changed_fields:
        changes_human = ", ".join(
            f"{field}={value!r} (was {item.get(field)!r})" for field, value in changed_fields.items()
        )
        logger.info("Updated product fields: %s", changes_human)
    else:
        if permitted_updates:
            unchanged_human = ", ".join(f"{k}={v!r}" for k, v in permitted_updates.items())
            logger.info(
                "No effective changes applied; values identical to existing: %s",
                unchanged_human,
            )
        else:
            logger.info(
                "No permitted updates to apply. Permitted fields are: %s",
                ", ".join(PERMITTED_FIELDS),
            )

    return merge_item_with_updates(item, permitted_updates)
