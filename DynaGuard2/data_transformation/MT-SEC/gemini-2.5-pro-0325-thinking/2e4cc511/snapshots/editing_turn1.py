def simple_update(product: dict, new_data: dict) -> dict:
    """
    Updates a product dictionary with new data.

    Args:
        product: The original product dictionary.
        new_data: A dictionary containing keys and values to update in the product.

    Returns:
        The updated product dictionary.
    """
    product.update(new_data)
    return product
