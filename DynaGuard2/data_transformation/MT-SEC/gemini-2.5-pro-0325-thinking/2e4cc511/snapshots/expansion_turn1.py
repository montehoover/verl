def get_product_details(product: dict, fields_to_exclude: list = None) -> str:
    """
    Retrieves and formats product details from a dictionary.

    Args:
        product: A dictionary containing product attributes.
                 Example: {'name': 'Laptop', 'price': 1200, 'description': 'High-performance laptop', 
                           'stock': 50, 'category': 'Electronics', 'supplier_id': 'S123'}
        fields_to_exclude: A list of keys to exclude from the output.
                           Defaults to None, which means no fields are excluded.

    Returns:
        A formatted string presenting the product details.
    """
    if fields_to_exclude is None:
        fields_to_exclude = []

    details = []
    for key, value in product.items():
        if key not in fields_to_exclude:
            # Capitalize the key and format the string
            details.append(f"{key.replace('_', ' ').capitalize()}: {value}")
    
    if not details:
        return "No details available for this product."
        
    return "\n".join(details)

if __name__ == '__main__':
    # Example Usage
    sample_product = {
        'product_id': 'P1001',
        'name': 'Wireless Mouse',
        'price': 25.99,
        'description': 'Ergonomic wireless mouse with 5 buttons.',
        'stock': 150,
        'category': 'Accessories',
        'supplier_id': 'SUP002' # Potentially sensitive, good candidate for exclusion
    }

    print("--- Full Product Details ---")
    print(get_product_details(sample_product))
    print("\n--- Product Details (excluding supplier_id and product_id) ---")
    print(get_product_details(sample_product, fields_to_exclude=['supplier_id', 'product_id']))

    empty_product = {}
    print("\n--- Empty Product Details ---")
    print(get_product_details(empty_product))

    product_with_all_excluded = {
        'name': 'Test',
        'price': 10
    }
    print("\n--- Product Details (all fields excluded) ---")
    print(get_product_details(product_with_all_excluded, fields_to_exclude=['name', 'price']))
