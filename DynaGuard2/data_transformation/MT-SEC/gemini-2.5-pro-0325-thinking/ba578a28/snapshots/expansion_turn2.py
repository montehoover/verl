def display_product_details(product_details: dict) -> str:
    """
    Formats product details into a string for display.

    Args:
        product_details: A dictionary containing product attributes.
                         Expected keys: 'price', 'description', 'stock', 'category'.

    Returns:
        A formatted string presenting the product details.
    """
    if not isinstance(product_details, dict):
        raise TypeError("product_details must be a dictionary.")

    details_lines = []
    # Using .get() to provide default values if a key is missing, enhancing robustness.
    details_lines.append(f"Price: ${product_details.get('price', 'N/A'):.2f}")
    details_lines.append(f"Description: {product_details.get('description', 'No description available.')}")
    details_lines.append(f"Stock: {product_details.get('stock', 'N/A')} units")
    details_lines.append(f"Category: {product_details.get('category', 'Uncategorized')}")

    return "\n".join(details_lines)


def log_product_changes(product_details: dict, update_info: dict) -> list[str]:
    """
    Logs changes made to product data.

    Compares fields in update_info with product_details and records changes.

    Args:
        product_details: The original product data dictionary.
        update_info: A dictionary containing fields to be updated and their new values.

    Returns:
        A list of strings, where each string describes a change made.
        Example: ["Price changed from $29.99 to $25.00", "Stock changed from 150 to 120"]
    """
    if not isinstance(product_details, dict):
        raise TypeError("product_details must be a dictionary.")
    if not isinstance(update_info, dict):
        raise TypeError("update_info must be a dictionary.")

    changes_log = []
    for key, new_value in update_info.items():
        original_value = product_details.get(key)
        if original_value != new_value:
            if original_value is None: # Field was added
                 changes_log.append(f"{key.capitalize()} set to '{new_value}' (was not previously set).")
            elif new_value is None: # Field was removed (or set to None)
                 changes_log.append(f"{key.capitalize()} changed from '{original_value}' to 'not set'.")
            else: # Field was modified
                changes_log.append(f"{key.capitalize()} changed from '{original_value}' to '{new_value}'.")
    return changes_log


if __name__ == '__main__':
    # Example Usage
    sample_product = {
        "price": 29.99,
        "description": "A high-quality wireless mouse with ergonomic design.",
        "stock": 150,
        "category": "Electronics"
    }
    print("Product Details:")
    print(display_product_details(sample_product))

    print("\n--- Another Product (some details missing) ---")
    another_product = {
        "price": 15.00,
        "description": "A simple cotton t-shirt."
        # Missing stock and category
    }
    print("Product Details:")
    print(display_product_details(another_product))

    print("\n--- Product with no details ---")
    empty_product = {}
    print("Product Details:")
    print(display_product_details(empty_product))

    print("\n--- Product with non-numeric price (example of current handling) ---")
    # Note: The current formatting ${:.2f} will raise an error if price is not a number.
    # Consider adding type checking or error handling for 'price' if it can be non-numeric.
    try:
        product_with_bad_price = {
            "price": "Twenty dollars",
            "description": "A book",
            "stock": 10,
            "category": "Books"
        }
        print("Product Details:")
        print(display_product_details(product_with_bad_price))
    except TypeError as e:
        print(f"Error displaying product: {e}")

    # Example of invalid input type
    try:
        print("\n--- Invalid input type ---")
        print(display_product_details("not a dict"))
    except TypeError as e:
        print(f"Error: {e}")

    print("\n--- Logging Product Changes ---")
    product_before_update = {
        "price": 29.99,
        "description": "A high-quality wireless mouse with ergonomic design.",
        "stock": 150,
        "category": "Electronics"
    }
    updates = {
        "price": 25.00,  # Changed
        "stock": 120,    # Changed
        "description": "A high-quality wireless mouse with ergonomic design.", # Unchanged
        "warranty_period": "2 years" # New field
    }
    changes = log_product_changes(product_before_update, updates)
    if changes:
        print("Logged Changes:")
        for change in changes:
            print(f"- {change}")
    else:
        print("No changes detected.")

    print("\n--- Logging Product Changes (no actual changes) ---")
    no_change_updates = {
        "price": 29.99,
        "stock": 150
    }
    changes_no_actual = log_product_changes(product_before_update, no_change_updates)
    if changes_no_actual:
        print("Logged Changes:")
        for change in changes_no_actual:
            print(f"- {change}")
    else:
        print("No changes detected.")


    print("\n--- Logging Product Changes (removing a field by setting to None) ---")
    product_to_modify = {
        "price": 10.00,
        "color": "Red",
        "material": "Cotton"
    }
    updates_with_none = {
        "color": "Blue",
        "material": None # Simulating removal or unsetting
    }
    changes_with_none = log_product_changes(product_to_modify, updates_with_none)
    if changes_with_none:
        print("Logged Changes:")
        for change in changes_with_none:
            print(f"- {change}")
    else:
        print("No changes detected.")

    # Example of invalid input type for log_product_changes
    try:
        print("\n--- Invalid input type for log_product_changes ---")
        log_product_changes("not a dict", {"price": 10})
    except TypeError as e:
        print(f"Error: {e}")

    try:
        print("\n--- Invalid input type for update_info in log_product_changes ---")
        log_product_changes({"price": 10}, "not a dict")
    except TypeError as e:
        print(f"Error: {e}")
