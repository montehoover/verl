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


def log_product_changes(product_info: dict, changes: dict) -> tuple[dict, list[str]]:
    """
    Updates product information with changes and logs these changes.

    Args:
        product_info: The original dictionary of product attributes.
        changes: A dictionary where keys are attribute names to be updated
                 and values are the new attribute values.

    Returns:
        A tuple containing:
            - The updated product_info dictionary.
            - A list of strings, where each string logs a change made.
    """
    change_log = []
    updated_product_info = product_info.copy()  # Work on a copy to avoid modifying the original dict directly if it's not desired

    for key, new_value in changes.items():
        old_value = updated_product_info.get(key, "N/A (new field)")
        if old_value != new_value:
            change_log.append(f"Changed '{key}': from '{old_value}' to '{new_value}'")
            updated_product_info[key] = new_value
        elif key not in updated_product_info: # Handles adding a new key explicitly
             change_log.append(f"Added '{key}': '{new_value}'")
             updated_product_info[key] = new_value


    return updated_product_info, change_log

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

    # Example Usage for log_product_changes
    print("\n--- Logging Product Changes ---")
    current_product_details = {
        'name': 'Laptop Pro',
        'price': 1500.00,
        'stock': 30,
        'category': 'Electronics'
    }
    print("Original Product Details:")
    print(display_product_info(current_product_details))

    changes_to_apply = {
        'price': 1450.00,  # Price decrease
        'stock': 25,       # Stock decrease
        'status': 'active' # New field
    }
    print("\nChanges to apply:")
    for k, v in changes_to_apply.items():
        print(f"- Update {k} to {v}")

    updated_product, logs = log_product_changes(current_product_details, changes_to_apply)

    print("\nUpdated Product Details:")
    print(display_product_info(updated_product))

    print("\nChange Log:")
    if logs:
        for log_entry in logs:
            print(log_entry)
    else:
        print("No changes were made.")

    # Example: No changes
    print("\n--- Logging Product Changes (No actual changes) ---")
    no_change_product = {'name': 'Tablet', 'price': 300}
    no_changes_to_apply = {'price': 300} # Same price
    
    updated_no_change_product, no_change_logs = log_product_changes(no_change_product, no_changes_to_apply)
    print("Original Product Details:")
    print(display_product_info(no_change_product))
    print("\nUpdated Product Details:")
    print(display_product_info(updated_no_change_product))
    print("\nChange Log:")
    if no_change_logs:
        for log_entry in no_change_logs:
            print(log_entry)
    else:
        print("No changes were made.")
    
    # Example: Changing an existing field and adding a new one
    print("\n--- Logging Product Changes (Mix of update and new field) ---")
    product_mix = {'name': 'Keyboard', 'color': 'Black'}
    changes_mix = {'color': 'White', 'type': 'Mechanical'}

    updated_product_mix, mix_logs = log_product_changes(product_mix, changes_mix)
    print("Original Product Details:")
    print(display_product_info(product_mix))
    print("\nUpdated Product Details:")
    print(display_product_info(updated_product_mix))
    print("\nChange Log:")
    for log_entry in mix_logs:
        print(log_entry)
