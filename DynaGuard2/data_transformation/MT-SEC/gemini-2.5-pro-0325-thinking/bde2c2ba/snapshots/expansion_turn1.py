def display_product_info(product_info: dict, exclude_fields: list = None) -> str:
    """
    Formats product information into a display string.

    Args:
        product_info: A dictionary containing product attributes.
                      Example: {'name': 'Laptop', 'price': 1200, 'description': 'High-performance laptop', 
                                'stock': 50, 'category': 'Electronics'}
        exclude_fields: An optional list of strings representing keys to exclude 
                        from the output. Example: ['category']

    Returns:
        A string with formatted product details.
    """
    if exclude_fields is None:
        exclude_fields = []

    display_items = []
    for key, value in product_info.items():
        if key not in exclude_fields:
            # Capitalize the key and replace underscores with spaces for better readability
            formatted_key = key.replace('_', ' ').capitalize()
            display_items.append(f"{formatted_key}: {value}")
    
    if not display_items:
        return "No product information to display."
        
    return "\n".join(display_items)

if __name__ == '__main__':
    # Example Usage
    sample_product = {
        'name': 'Wireless Mouse',
        'price': 25.99,
        'description': 'Ergonomic wireless mouse with 2.4 GHz connectivity.',
        'stock': 150,
        'category': 'Accessories',
        'product_id': 'WM-001'
    }

    print("--- Full Product Details ---")
    print(display_product_info(sample_product))

    print("\n--- Product Details (excluding 'category' and 'product_id') ---")
    print(display_product_info(sample_product, exclude_fields=['category', 'product_id']))

    print("\n--- Product Details (excluding 'description') ---")
    print(display_product_info(sample_product, exclude_fields=['description']))

    empty_product = {}
    print("\n--- Empty Product Details ---")
    print(display_product_info(empty_product))

    product_with_all_excluded = {
        'name': 'Test Product',
        'price': 10.00
    }
    print("\n--- Product Details (all fields excluded) ---")
    print(display_product_info(product_with_all_excluded, exclude_fields=['name', 'price']))
