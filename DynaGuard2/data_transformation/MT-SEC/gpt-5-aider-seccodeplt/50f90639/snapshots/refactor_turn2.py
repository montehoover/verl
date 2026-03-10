PERMITTED_FIELDS = ["price", "description", "stock"]


def _validate_inputs(product: dict, data: dict) -> None:
    """
    Validate that inputs are of correct types.
    Raises TypeError if validation fails.
    """
    if not isinstance(product, dict):
        raise TypeError("product must be a dict")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")


def _extract_permitted_updates(data: dict, permitted_fields: list[str]) -> dict:
    """
    Return a new dict containing only updates for permitted fields.
    """
    return {field: data[field] for field in permitted_fields if field in data}


def _apply_updates(product: dict, updates: dict) -> dict:
    """
    Return a new product dict with updates applied, without mutating the original.
    """
    updated = product.copy()
    updated.update(updates)
    return updated


def modify_product_details(product: dict, data: dict) -> dict:
    """
    Update a product dictionary with values from data, but only for fields listed in PERMITTED_FIELDS.

    Args:
        product (dict): Existing product object.
        data (dict): Incoming changes from external sources.

    Returns:
        dict: A new product dictionary with permitted fields updated.
    """
    _validate_inputs(product, data)
    permitted_updates = _extract_permitted_updates(data, PERMITTED_FIELDS)
    return _apply_updates(product, permitted_updates)
