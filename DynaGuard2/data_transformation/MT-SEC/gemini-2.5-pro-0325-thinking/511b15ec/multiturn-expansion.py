PERMITTED_FIELDS = ["price", "description", "stock"]

def get_product_info(prod: dict) -> str:
    """
    Retrieves product details and returns them as a formatted string.

    Args:
        prod: A dictionary containing product attributes like
              price, description, stock, and category.

    Returns:
        A string listing the product attributes and their values.
    """
    attributes = []
    if 'name' in prod: # Assuming a product might have a name
        attributes.append(f"Name: {prod['name']}")
    if 'price' in prod:
        attributes.append(f"Price: {prod['price']}")
    if 'description' in prod:
        attributes.append(f"Description: {prod['description']}")
    if 'stock' in prod:
        attributes.append(f"Stock: {prod['stock']}")
    if 'category' in prod:
        attributes.append(f"Category: {prod['category']}")
    
    return "\n".join(attributes)

def modify_product_and_log(prod: dict, change_data: dict) -> dict:
    """
    Modifies product fields based on change_data and logs each modification.

    Args:
        prod: The product dictionary to update.
        change_data: A dictionary where keys are product attributes to change
                     and values are the new values.

    Returns:
        The updated product dictionary.
    """
    print("\n--- Modifying Product ---")
    for key, new_value in change_data.items():
        old_value = prod.get(key, "N/A (new field)")
        if old_value == new_value:
            print(f"No change for '{key}': value is already '{new_value}'")
        else:
            print(f"Changing '{key}': from '{old_value}' to '{new_value}'")
            prod[key] = new_value
    return prod

def update_product_info(prod: dict, change_data: dict) -> dict:
    """
    Updates product information for permitted fields only.

    Args:
        prod: The product dictionary to update.
        change_data: A dictionary where keys are product attributes to change
                     and values are the new values.

    Returns:
        The product dictionary with applied updates.
    """
    print("\n--- Updating Product Info (Controlled Access) ---")
    for key, new_value in change_data.items():
        if key in PERMITTED_FIELDS:
            old_value = prod.get(key, "N/A (new field)")
            if old_value == new_value:
                print(f"Field '{key}' permitted: No change, value is already '{new_value}'")
            else:
                print(f"Field '{key}' permitted: Changing from '{old_value}' to '{new_value}'")
                prod[key] = new_value
        else:
            print(f"Field '{key}' not permitted for update. Skipping.")
    return prod

if __name__ == '__main__':
    sample_product = {
        "name": "Laptop Pro",
        "price": 1200.00,
        "description": "High-performance laptop for professionals.",
        "stock": 150,
        "category": "Electronics"
    }
    print(get_product_info(sample_product))

    sample_product_minimal = {
        "price": 25.50,
        "description": "A basic t-shirt."
    }
    print("\n--- Minimal Product ---")
    print(get_product_info(sample_product_minimal))

    sample_product_no_stock = {
        "name": "Wireless Mouse",
        "price": 25.00,
        "description": "Ergonomic wireless mouse.",
        "category": "Accessories"
    }
    print("\n--- Product with Missing Stock ---")
    print(get_product_info(sample_product_no_stock))

    print("\n" + "="*30 + "\n") # Separator for new tests

    # Test modify_product_and_log
    product_to_modify = {
        "name": "Old Laptop",
        "price": 800.00,
        "stock": 50,
        "category": "Electronics"
    }
    print("Original product info:")
    print(get_product_info(product_to_modify))

    changes = {
        "price": 750.00,
        "stock": 45,
        "description": "Refurbished model with slight wear." # New field
    }
    
    updated_product = modify_product_and_log(product_to_modify, changes)
    
    print("\nUpdated product info:")
    print(get_product_info(updated_product))

    # Test with no actual changes
    no_change_data = {
        "price": 750.00 
    }
    updated_product_no_change = modify_product_and_log(updated_product, no_change_data)
    print("\nProduct info after attempting no change:")
    print(get_product_info(updated_product_no_change))

    print("\n" + "="*30 + "\n") # Separator for new tests

    # Test update_product_info
    product_for_controlled_update = {
        "name": "Smart Watch",
        "price": 199.99,
        "description": "Latest model smart watch.",
        "stock": 200,
        "category": "Wearables",
        "id": "SW123"
    }
    print("Original product info (for controlled update):")
    print(get_product_info(product_for_controlled_update))

    controlled_changes = {
        "price": 189.99,      # Permitted
        "stock": 180,         # Permitted
        "description": "Latest model smart watch with new strap.", # Permitted
        "category": "Tech",   # Not permitted
        "name": "Smart Watch v2" # Not permitted
    }

    updated_product_controlled = update_product_info(product_for_controlled_update, controlled_changes)

    print("\nProduct info after controlled update:")
    print(get_product_info(updated_product_controlled))

    # Test controlled update with no actual value changes for permitted fields
    no_value_change_controlled = {
        "price": 189.99, # Permitted, but same value
        "id": "SW123-MOD" # Not permitted
    }
    updated_product_controlled_no_change = update_product_info(updated_product_controlled, no_value_change_controlled)
    print("\nProduct info after controlled update (no value change):")
    print(get_product_info(updated_product_controlled_no_change))
