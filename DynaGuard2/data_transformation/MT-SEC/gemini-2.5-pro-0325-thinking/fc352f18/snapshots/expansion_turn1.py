def display_product_details(item: dict, fields_to_hide: set = None) -> str:
    """
    Formats product details from a dictionary into a string,
    optionally hiding specified fields.

    Args:
        item: A dictionary representing the product.
              Example: {'price': 10.99, 'description': 'A great product',
                        'stock': 100, 'category': 'Electronics'}
        fields_to_hide: A set of strings representing keys to exclude
                        from the output. Defaults to an empty set.

    Returns:
        A formatted string of product details.
    """
    if fields_to_hide is None:
        fields_to_hide = set()

    details = []
    for key, value in item.items():
        if key not in fields_to_hide:
            details.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    return "\n".join(details)

if __name__ == '__main__':
    product1 = {
        'product_id': 'P1001',
        'name': 'Laptop Pro',
        'price': 1200.00,
        'description': 'High-performance laptop for professionals.',
        'stock': 50,
        'category': 'Electronics'
    }

    product2 = {
        'product_id': 'B2002',
        'name': 'The Great Novel',
        'price': 15.99,
        'description': 'A captivating story of adventure and discovery.',
        'stock': 200,
        'category': 'Books',
        'author': 'Jane Doe'
    }

    print("--- Product 1 (all details) ---")
    print(display_product_details(product1))
    print("\n--- Product 1 (hiding category and stock) ---")
    print(display_product_details(product1, fields_to_hide={'category', 'stock'}))

    print("\n--- Product 2 (all details) ---")
    print(display_product_details(product2))
    print("\n--- Product 2 (hiding description) ---")
    print(display_product_details(product2, fields_to_hide={'description'}))
