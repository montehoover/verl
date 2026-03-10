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
