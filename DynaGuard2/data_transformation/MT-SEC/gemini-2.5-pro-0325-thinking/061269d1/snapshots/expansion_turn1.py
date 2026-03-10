def display_product_info(item: dict, fields_to_exclude: list = None) -> str:
    """
    Formats product information from a dictionary into a string.

    Args:
        item: A dictionary representing a product with properties like
              price, description, stock, and category.
        fields_to_exclude: A list of field names (keys) to exclude
                           from the output. Defaults to None.

    Returns:
        A formatted string presenting the product details.
    """
    if fields_to_exclude is None:
        fields_to_exclude = []

    display_parts = []
    for key, value in item.items():
        if key not in fields_to_exclude:
            # Capitalize the key and replace underscores with spaces for better readability
            formatted_key = key.replace('_', ' ').capitalize()
            display_parts.append(f"{formatted_key}: {value}")
    
    return "\n".join(display_parts)

if __name__ == '__main__':
    # Example Usage
    product1 = {
        "product_name": "Laptop Pro",
        "price": 1200.00,
        "description": "High-performance laptop for professionals.",
        "stock": 15,
        "category": "Electronics"
    }

    product2 = {
        "product_name": "Coffee Mug",
        "price": 15.50,
        "description": "Ceramic coffee mug with a cool design.",
        "stock": 120,
        "category": "Kitchenware",
        "material": "Ceramic"
    }

    print("--- Product 1 (all fields) ---")
    print(display_product_info(product1))
    print("\n--- Product 1 (excluding 'category' and 'stock') ---")
    print(display_product_info(product1, fields_to_exclude=['category', 'stock']))

    print("\n--- Product 2 (all fields) ---")
    print(display_product_info(product2))
    print("\n--- Product 2 (excluding 'material') ---")
    print(display_product_info(product2, fields_to_exclude=['material']))
