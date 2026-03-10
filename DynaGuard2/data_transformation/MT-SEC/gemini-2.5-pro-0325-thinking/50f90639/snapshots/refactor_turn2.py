PERMITTED_FIELDS = ["price", "description", "stock"]

def _is_field_permitted(field_name: str) -> bool:
    """Checks if a field is permitted for update."""
    return field_name in PERMITTED_FIELDS

def _update_product_field(product: dict, field_name: str, value: any) -> dict:
    """Updates a single field in the product dictionary and returns a new dictionary."""
    if not isinstance(product, dict):
        raise TypeError("Product must be a dictionary.")
    updated_product = product.copy()
    updated_product[field_name] = value
    return updated_product

def modify_product_details(product: dict, data: dict) -> dict:
    """
    Updates product details based on input data provided by external sources.

    Args:
        product: dict, a dictionary representing the product object with specific fields.
        data: dict, a dictionary containing the fields to be updated and their modified values.

    Returns:
        A dictionary representing the modified product object.
    """
    current_product_state = product.copy()  # Avoid modifying the original product dict directly
    for key, value in data.items():
        if _is_field_permitted(key):
            current_product_state = _update_product_field(current_product_state, key, value)
    return current_product_state
