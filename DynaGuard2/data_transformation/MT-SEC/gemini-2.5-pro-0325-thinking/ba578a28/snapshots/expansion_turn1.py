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
