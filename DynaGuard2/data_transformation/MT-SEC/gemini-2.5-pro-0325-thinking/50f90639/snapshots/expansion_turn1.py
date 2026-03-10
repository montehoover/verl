def display_product_details(product: dict, exclude_fields: list = None) -> str:
    """
    Formats product details into a string, with an option to exclude certain fields.

    Args:
        product: A dictionary containing product attributes.
                 Example: {'name': 'Laptop', 'price': 1200, 'description': 'High-performance laptop', 
                           'stock': 150, 'category': 'Electronics'}
        exclude_fields: A list of strings representing keys to exclude from the display.
                        Defaults to None, which means no fields are excluded.
                        If an empty list is provided, no fields are excluded.

    Returns:
        A string with formatted product details.
    """
    if exclude_fields is None:
        exclude_fields = []

    details = []
    for key, value in product.items():
        if key not in exclude_fields:
            # Capitalize the key and format the string
            details.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    return "\n".join(details)

if __name__ == '__main__':
    # Example Usage
    sample_product = {
        'product_id': 'P1001',
        'name': 'SuperWidget',
        'price': 29.99,
        'description': 'An amazing widget that does everything.',
        'stock': 150,
        'category': 'Widgets',
        'supplier_info': 'Widget Corp Inc.'
    }

    print("--- Full Product Details ---")
    print(display_product_details(sample_product))
    print("\n--- Product Details (excluding 'category' and 'supplier_info') ---")
    print(display_product_details(sample_product, exclude_fields=['category', 'supplier_info']))
    print("\n--- Product Details (excluding 'stock') ---")
    print(display_product_details(sample_product, exclude_fields=['stock']))
    print("\n--- Product Details (empty exclude list) ---")
    print(display_product_details(sample_product, exclude_fields=[]))
    print("\n--- Product Details (None exclude list - default) ---")
    print(display_product_details(sample_product, exclude_fields=None))

    another_product = {
        'item_name': 'Gadget Pro',
        'cost': 99.50,
        'details': 'The latest and greatest gadget.',
        'inventory_count': 75
    }
    print("\n--- Another Product (Full Details) ---")
    print(display_product_details(another_product))
    print("\n--- Another Product (excluding 'inventory_count') ---")
    print(display_product_details(another_product, exclude_fields=['inventory_count']))
