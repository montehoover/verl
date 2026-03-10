import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
def log_product_changes(product: dict, changes: dict) -> dict:
    """
    Logs changes made to a product and updates the product dictionary.

    Args:
        product: The original product dictionary.
        changes: A dictionary containing the changes to apply. 
                 Keys are field names, values are new field values.

    Returns:
        The updated product dictionary.
    """
    updated_product = product.copy() # Work on a copy to avoid modifying the original dict directly if passed by reference elsewhere
    
    for key, new_value in changes.items():
        original_value = updated_product.get(key)
        if original_value != new_value: # Log only if there's an actual change or it's a new field
            if original_value is None: # New field being added
                 logging.info(f"Product field '{key}' added. New value: '{new_value}'.")
            else: # Existing field being updated
                logging.info(f"Product field '{key}' changed. Original: '{original_value}', New: '{new_value}'.")
            updated_product[key] = new_value
        # If original_value == new_value, no change, so no log and no update needed for this key.
            
    return updated_product

if __name__ == '__main__':

    # --- Test log_product_changes ---
    print("\n--- Logging Product Changes ---")
    product_to_update = {
        'product_id': 'P1002',
        'name': 'Gaming Keyboard',
        'price': 75.00,
        'stock': 50,
        'description': 'Mechanical gaming keyboard with RGB lighting.'
    }
    changes_to_apply = {
        'price': 79.99,  # Price increase
        'stock': 45,     # Stock decrease
        'description': 'Upgraded mechanical gaming keyboard with customizable RGB lighting.', # Description update
        'warranty_period': '2 years' # New field
    }
    
    print(f"Original product: {product_to_update}")
    updated_product = log_product_changes(product_to_update, changes_to_apply)
    print(f"Updated product: {updated_product}")

    print("\n--- Product Details After Update (excluding product_id) ---")
    print(get_product_details(updated_product, fields_to_exclude=['product_id']))

    print("\n--- Logging Changes for a new product (all fields are new) ---")
    new_product = {}
    new_product_data = {
        'product_id': 'P1003',
        'name': 'Webcam',
        'price': 45.00,
        'stock': 30
    }
    updated_new_product = log_product_changes(new_product, new_product_data)
    print(f"Updated new product: {updated_new_product}")
