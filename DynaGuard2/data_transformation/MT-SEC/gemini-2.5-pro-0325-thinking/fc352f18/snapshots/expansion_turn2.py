import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def log_and_update_product(item: dict, payload: dict) -> dict:
    """
    Updates product details from a payload and logs changes.

    Args:
        item: The original product dictionary.
        payload: A dictionary with keys to update and their new values.

    Returns:
        The updated product dictionary.
    """
    updated_item = item.copy()  # Work on a copy to avoid modifying the original dict directly if not intended
    logging.info(f"Updating product ID: {updated_item.get('product_id', 'N/A')}")
    for key, new_value in payload.items():
        if key in updated_item:
            old_value = updated_item[key]
            if old_value != new_value:
                logging.info(f"Changing '{key}': from '{old_value}' to '{new_value}'")
                updated_item[key] = new_value
            else:
                logging.info(f"No change for '{key}': value is already '{new_value}'")
        else:
            logging.warning(f"Key '{key}' not found in product, cannot update.")
    return updated_item

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

    print("\n--- Updating Product 1 ---")
    update_payload1 = {
        'price': 1150.00,
        'stock': 45,
        'description': 'High-performance laptop for professionals and gamers.'
    }
    updated_product1 = log_and_update_product(product1, update_payload1)
    print("\n--- Updated Product 1 details ---")
    print(display_product_details(updated_product1))

    print("\n--- Attempting to update Product 2 with a non-existent field ---")
    update_payload2 = {
        'price': 14.99,
        'non_existent_field': 'some_value'
    }
    updated_product2 = log_and_update_product(product2, update_payload2)
    print("\n--- Updated Product 2 details ---")
    print(display_product_details(updated_product2))
