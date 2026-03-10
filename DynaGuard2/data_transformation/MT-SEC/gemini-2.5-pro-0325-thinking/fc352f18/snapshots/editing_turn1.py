def update_product_info(product: dict, new_data: dict) -> dict:
    """
    Updates the product dictionary with new data.

    Args:
        product: The original product dictionary.
        new_data: A dictionary containing keys and values to update in the product.

    Returns:
        The updated product dictionary.
    """
    updated_product = product.copy()
    updated_product.update(new_data)
    return updated_product
