import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PERMITTED_FIELDS = ["price", "description", "stock"]

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

def log_product_changes(item: dict, details: dict) -> dict:
    """
    Updates product attributes based on the details dictionary and logs changes.

    Args:
        item: The product dictionary to be updated.
        details: A dictionary containing the proposed changes (attribute: new_value).

    Returns:
        The updated product dictionary.
    """
    updated_item = item.copy()  # Work on a copy to avoid modifying the original dict directly if not intended
    product_name = updated_item.get("product_name", "Unknown Product")

    for key, new_value in details.items():
        if key in updated_item:
            original_value = updated_item[key]
            if original_value != new_value:
                logging.info(f"Product '{product_name}': Changed '{key}' from '{original_value}' to '{new_value}'")
                updated_item[key] = new_value
            else:
                logging.info(f"Product '{product_name}': No change for '{key}', value is already '{new_value}'")
        else:
            logging.warning(f"Product '{product_name}': Attribute '{key}' not found in product, cannot update.")
            # Optionally, you could add the new key-value pair if that's desired behavior
            # updated_item[key] = new_value
            # logging.info(f"Product '{product_name}': Added new attribute '{key}' with value '{new_value}'")


    return updated_item

def edit_product_info(item: dict, details: dict) -> dict:
    """
    Updates product information for permitted fields and logs changes.

    Args:
        item: The product dictionary to be updated.
        details: A dictionary containing the proposed changes (attribute: new_value).
                 Only attributes listed in PERMITTED_FIELDS will be considered.

    Returns:
        The updated product dictionary.
    """
    updated_item = item.copy()
    product_name = updated_item.get("product_name", "Unknown Product")

    for key, new_value in details.items():
        if key in PERMITTED_FIELDS:
            if key in updated_item:
                original_value = updated_item[key]
                if original_value != new_value:
                    logging.info(f"Product '{product_name}' (edit_product_info): Changed '{key}' from '{original_value}' to '{new_value}'")
                    updated_item[key] = new_value
                else:
                    logging.info(f"Product '{product_name}' (edit_product_info): No change for '{key}', value is already '{new_value}'")
            else:
                # This case should ideally not happen if item structure is consistent
                # and PERMITTED_FIELDS are actual attributes of the item.
                logging.warning(f"Product '{product_name}' (edit_product_info): Permitted attribute '{key}' not found in product, cannot update.")
        else:
            logging.info(f"Product '{product_name}' (edit_product_info): Attribute '{key}' is not permitted for direct editing or not found.")
    
    return updated_item

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

    print("\n--- Logging Product Changes ---")
    changes_to_product1 = {
        "price": 1150.00,
        "stock": 10,
        "description": "High-performance laptop for professionals and gamers." # Changed
    }
    print(f"\nOriginal Product 1: {product1}")
    updated_product1 = log_product_changes(product1, changes_to_product1)
    print(f"Updated Product 1: {updated_product1}")
    print("\n--- Displaying Updated Product 1 ---")
    print(display_product_info(updated_product1))

    # Example of trying to change a non-existent field and a field with the same value
    changes_to_product2 = {
        "price": 15.50, # Same value
        "stock": 100,
        "color": "Blue" # New field
    }
    print(f"\nOriginal Product 2: {product2}")
    updated_product2 = log_product_changes(product2, changes_to_product2)
    print(f"Updated Product 2: {updated_product2}")
    print("\n--- Displaying Updated Product 2 ---")
    print(display_product_info(updated_product2))

    print("\n--- Editing Product Info (with PERMITTED_FIELDS) ---")
    product3 = {
        "product_name": "Wireless Mouse",
        "price": 25.00,
        "description": "Ergonomic wireless mouse.",
        "stock": 200,
        "category": "Accessories",
        "color": "Black"
    }
    print(f"\nOriginal Product 3: {product3}")
    print(display_product_info(product3))

    edit_details_product3 = {
        "price": 22.50,  # Permitted
        "description": "Ergonomic wireless mouse with adjustable DPI.", # Permitted
        "stock": 180,    # Permitted
        "category": "Computer Accessories", # Not permitted
        "color": "Silver" # Not permitted
    }
    
    updated_product3 = edit_product_info(product3, edit_details_product3)
    print(f"\nUpdated Product 3 after edit_product_info: {updated_product3}")
    print("\n--- Displaying Updated Product 3 ---")
    print(display_product_info(updated_product3))

    # Example: Trying to edit a field that is permitted but not in the product
    # (This scenario is less likely if PERMITTED_FIELDS are well-defined based on product structure)
    product4 = {"product_name": "Keyboard", "price": 75.00} # No 'description' or 'stock'
    edit_details_product4 = {
        "price": 70.00,
        "description": "Mechanical Keyboard", # Permitted, but not in product4 initially
        "stock": 50 # Permitted, but not in product4 initially
    }
    print(f"\nOriginal Product 4: {product4}")
    updated_product4 = edit_product_info(product4, edit_details_product4)
    print(f"\nUpdated Product 4 after edit_product_info: {updated_product4}")
    print(display_product_info(updated_product4))
