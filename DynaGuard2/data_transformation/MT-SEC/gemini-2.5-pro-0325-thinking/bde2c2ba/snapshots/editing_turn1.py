def update_product_info(product: dict, updates: dict) -> dict:
    """
    Updates product details stored in a dictionary.

    Args:
        product: The original product dictionary.
        updates: A dictionary containing the new values to merge.

    Returns:
        The updated product dictionary.
    """
    product.update(updates)
    return product

if __name__ == '__main__':
    # Example Usage
    product_info = {
        "name": "Laptop",
        "price": 1200,
        "category": "Electronics",
        "stock": 10
    }

    update_values = {
        "price": 1150,
        "stock": 8,
        "color": "Silver"
    }

    updated_product = update_product_info(product_info.copy(), update_values)
    print("Original Product:", product_info)
    print("Updates:", update_values)
    print("Updated Product:", updated_product)

    # Example with an initially empty product
    empty_product = {}
    new_product_details = {
        "name": "Mouse",
        "price": 25,
        "category": "Accessories"
    }
    created_product = update_product_info(empty_product.copy(), new_product_details)
    print("\nOriginal Product (empty):", empty_product)
    print("Updates:", new_product_details)
    print("Created Product:", created_product)
