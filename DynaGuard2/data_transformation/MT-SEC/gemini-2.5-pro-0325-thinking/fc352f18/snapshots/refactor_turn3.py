PERMITTED_FIELDS = ["price", "description", "stock"]  # Global list of fields allowed for modification

def _filter_permitted_payload(payload: dict, permitted_fields: list) -> dict:
    """
    Filters the incoming payload dictionary to include only those keys
    that are present in the `permitted_fields` list.

    Args:
        payload: A dictionary containing the data to update the product with.
                 Example: {"price": 99.99, "category": "electronics"}
        permitted_fields: A list of strings representing field names that are
                          allowed to be updated.
                          Example: ["price", "description", "stock"]

    Returns:
        A new dictionary containing only the key-value pairs from the payload
        where the key is in `permitted_fields`.
        Example: {"price": 99.99}
    """
    # Use a dictionary comprehension for a concise way to build the filtered dictionary.
    # It iterates through each item in the payload.
    # If the item's key is found in permitted_fields, it's included in the new dictionary.
    return {
        key: value for key, value in payload.items() if key in permitted_fields
    }

def _apply_updates(item: dict, updates: dict) -> dict:
    """
    Applies the given updates to a copy of the item dictionary.

    This function ensures that the original item dictionary is not modified directly,
    promoting immutability for the input `item`.

    Args:
        item: The original product dictionary.
              Example: {"name": "Laptop", "price": 1200.00, "stock": 10}
        updates: A dictionary containing the field updates to apply.
                 These updates should already be validated (e.g., filtered by
                 `_filter_permitted_payload`).
                 Example: {"price": 1150.00, "stock": 8}

    Returns:
        A new dictionary representing the item with the updates applied.
        Example: {"name": "Laptop", "price": 1150.00, "stock": 8}
    """
    # Create a shallow copy of the original item to avoid modifying it directly.
    # This is important if the caller expects the original item to remain unchanged.
    updated_item = item.copy()
    # The update() method merges the `updates` dictionary into `updated_item`.
    # If a key exists in both dictionaries, its value in `updated_item` is
    # replaced by its value in `updates`.
    updated_item.update(updates)
    return updated_item

def amend_product_features(item: dict, payload: dict) -> dict:
    """
    Updates specified fields of a product dictionary using incoming payload data.

    This function orchestrates the update process:
    1. It filters the `payload` to ensure only permitted fields are considered for update,
       based on the globally defined `PERMITTED_FIELDS` list.
    2. It applies these filtered updates to a copy of the `item` dictionary.

    Args:
        item: dict, the dictionary representing the product object with its
              current fields.
              Example: {"id": 1, "name": "Teapot", "price": 25.00, "description": "A sturdy teapot.", "stock": 100}
        payload: dict, a dictionary containing the new values for the fields
                 that need to be updated. Fields not in `PERMITTED_FIELDS`
                 or not present in the payload will be ignored or left unchanged.
                 Example: {"price": 23.50, "stock": 95, "color": "red"}

    Returns:
        A dictionary reflecting the product object with the changes applied.
        Only fields present in `PERMITTED_FIELDS` and the `payload` are updated.
        Example (assuming "color" is not in PERMITTED_FIELDS):
        {"id": 1, "name": "Teapot", "price": 23.50, "description": "A sturdy teapot.", "stock": 95}
    """
    # Step 1: Filter the incoming payload to get only the data for fields that are allowed to be changed.
    # This prevents unauthorized or unintended modifications to the product data.
    valid_payload = _filter_permitted_payload(payload, PERMITTED_FIELDS)

    # Step 2: Apply the validated and filtered updates to the product item.
    # This operation returns a new dictionary with the updates, leaving the original 'item' unchanged.
    updated_item = _apply_updates(item, valid_payload)

    return updated_item
