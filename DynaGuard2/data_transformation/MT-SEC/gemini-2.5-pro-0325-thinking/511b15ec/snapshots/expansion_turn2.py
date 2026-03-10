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
