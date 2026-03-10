PERMITTED_FIELDS = ['price', 'description', 'stock']

def update_item_information(product_info: dict, new_data: dict) -> dict:
    """
    Updates product information for specific fields only.

    Args:
        product_info: The original product dictionary.
        new_data: A dictionary containing the new values to merge.

    Returns:
        The updated product dictionary with only permitted fields updated.
    """
    updated_product = product_info.copy()
    for key, value in new_data.items():
        if key in PERMITTED_FIELDS:
            updated_product[key] = value
    return updated_product


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


def restricted_update(product: dict, updates: dict, allowed_fields: list[str]) -> dict:
    """
    Updates product details stored in a dictionary, but only for allowed fields.

    Args:
        product: The original product dictionary.
        updates: A dictionary containing the new values to merge.
        allowed_fields: A list of strings representing fields that are allowed to be updated.

    Returns:
        The updated product dictionary.
    """
    for key, value in updates.items():
        if key in allowed_fields:
            product[key] = value
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

    # Example Usage for restricted_update
    product_info_for_restricted_update = {
        "name": "Tablet",
        "price": 300,
        "category": "Electronics",
        "stock": 15,
        "brand": "BrandX"
    }

    updates_for_restricted = {
        "price": 280,  # Allowed
        "stock": 12,  # Allowed
        "color": "Black",  # Not allowed
        "brand": "BrandY" # Not allowed
    }

    allowed_to_update = ["price", "stock"]

    updated_product_restricted = restricted_update(
        product_info_for_restricted_update.copy(),
        updates_for_restricted,
        allowed_to_update
    )
    print("\nOriginal Product (for restricted update):", product_info_for_restricted_update)
    print("Updates (for restricted update):", updates_for_restricted)
    print("Allowed Fields:", allowed_to_update)
    print("Updated Product (restricted):", updated_product_restricted)

    # Example Usage for update_item_information
    item_details = {
        "name": "Smartphone",
        "price": 700,
        "description": "Latest model smartphone",
        "stock": 20,
        "category": "Electronics"
    }

    data_to_update = {
        "price": 650,  # Permitted
        "description": "Latest model smartphone with improved camera",  # Permitted
        "stock": 18,  # Permitted
        "category": "Mobile Devices",  # Not permitted
        "manufacturer": "TechCorp" # Not permitted
    }

    updated_item = update_item_information(item_details.copy(), data_to_update)
    print("\nOriginal Item Details:", item_details)
    print("Data to Update:", data_to_update)
    print("Permitted Fields for update_item_information:", PERMITTED_FIELDS)
    print("Updated Item Details:", updated_item)
