# Global list of fields that are allowed to be updated in a product.
PERMITTED_FIELDS = ["price", "description", "stock"]


def _is_field_permitted(field_name: str) -> bool:
    """
    Checks if a given field name is in the list of permitted fields.

    Args:
        field_name: The name of the field to check.

    Returns:
        True if the field is permitted, False otherwise.
    """
    return field_name in PERMITTED_FIELDS


def _update_product_field(product: dict, field_name: str, value: any) -> dict:
    """
    Updates a single field in the product dictionary.

    This function creates a shallow copy of the product dictionary to avoid
    modifying the original dictionary in place, ensuring immutability.

    Args:
        product: The product dictionary to update.
        field_name: The name of the field to update.
        value: The new value for the field.

    Returns:
        A new product dictionary with the specified field updated.

    Raises:
        TypeError: If the provided product is not a dictionary.
    """
    if not isinstance(product, dict):
        raise TypeError("Product must be a dictionary.")
    
    # Create a shallow copy to avoid modifying the original product dictionary.
    updated_product = product.copy()
    updated_product[field_name] = value
    return updated_product


def modify_product_details(product: dict, data: dict) -> dict:
    """
    Updates product details based on input data from external sources.

    This function iterates through the provided data. For each field in the data,
    it checks if the field is permitted for update. If it is, the product's
    field is updated with the new value. The function ensures that only
    fields listed in PERMITTED_FIELDS are modified.

    Args:
        product: A dictionary representing the product object.
                 Example: {'id': 1, 'name': 'Laptop', 'price': 1200.00, 'stock': 50}
        data: A dictionary containing the fields to be updated and their
              new values.
              Example: {'price': 1150.00, 'stock': 45, 'category': 'electronics'}
              Note: 'category' would be ignored if not in PERMITTED_FIELDS.

    Returns:
        A new dictionary representing the modified product object.
        If no fields in `data` are permitted or `data` is empty,
        a copy of the original `product` is returned.
    """
    # Start with a copy of the product to ensure the original is not mutated.
    current_product_state = product.copy()

    # Iterate over the key-value pairs in the input data.
    for key, value in data.items():
        # Check if the current field is allowed to be modified.
        if _is_field_permitted(key):
            # If permitted, update the field in our working copy of the product.
            current_product_state = _update_product_field(current_product_state, key, value)
            
    return current_product_state
