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


def log_product_changes(product: dict, changes: dict) -> tuple[dict, list[str]]:
    """
    Updates product details with new values and logs the changes.

    Args:
        product: The original product dictionary.
        changes: A dictionary containing the updates to apply.
                 Example: {'price': 1250, 'stock': 140}

    Returns:
        A tuple containing:
            - The updated product dictionary.
            - A list of strings, where each string logs a change.
              Example: ["Price changed from 1200 to 1250", "Stock changed from 150 to 140"]
    """
    change_log = []
    updated_product = product.copy()  # Work on a copy to avoid modifying the original dict directly if passed by reference elsewhere

    for key, new_value in changes.items():
        old_value = updated_product.get(key, 'N/A (New Field)') # Handle new fields being added
        if old_value != new_value:
            change_log.append(f"{key.replace('_', ' ').capitalize()} changed from {old_value} to {new_value}")
            updated_product[key] = new_value
        elif key not in updated_product: # If key was not in product and new_value is the default from get
             change_log.append(f"{key.replace('_', ' ').capitalize()} added with value {new_value}")
             updated_product[key] = new_value


    return updated_product, change_log

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

    print("\n\n--- Logging Product Changes ---")
    product_to_update = {
        'product_id': 'P2002',
        'name': 'EcoMug',
        'price': 15.00,
        'description': 'A reusable and eco-friendly mug.',
        'stock': 200,
        'category': 'Kitchenware'
    }
    changes_to_apply = {
        'price': 18.50,  # Update price
        'stock': 175,    # Update stock
        'material': 'Bamboo Fiber' # Add new field
    }

    print("Original Product:")
    print(display_product_details(product_to_update))
    
    updated_product, logs = log_product_changes(product_to_update, changes_to_apply)
    
    print("\nChanges Logged:")
    for log_entry in logs:
        print(log_entry)
        
    print("\nUpdated Product:")
    print(display_product_details(updated_product))

    # Example with no actual changes
    print("\n--- Logging Product Changes (No actual change) ---")
    no_change_product = {'name': 'Test', 'price': 10}
    no_changes_to_apply = {'price': 10}
    updated_no_change_product, no_change_logs = log_product_changes(no_change_product, no_changes_to_apply)
    print("Original Product:")
    print(display_product_details(no_change_product))
    print("\nChanges Logged:")
    if not no_change_logs:
        print("No changes were made.")
    else:
        for log_entry in no_change_logs:
            print(log_entry)
    print("\nUpdated Product:")
    print(display_product_details(updated_no_change_product))

    # Example adding a new field where old value is N/A
    print("\n--- Logging Product Changes (Adding new field only) ---")
    new_field_product = {'name': 'Test'}
    new_field_to_add = {'status': 'active'}
    updated_new_field_product, new_field_logs = log_product_changes(new_field_product, new_field_to_add)
    print("Original Product:")
    print(display_product_details(new_field_product))
    print("\nChanges Logged:")
    for log_entry in new_field_logs:
        print(log_entry)
    print("\nUpdated Product:")
    print(display_product_details(updated_new_field_product))
