# Predefined orders dictionary (can be expanded or moved to a database)
ORDERS = {
    "order123": {"customer_name": "Alice Smith", "customer_email": "alice@example.com", "items": ["itemA", "itemB"]},
    "order456": {"customer_name": "Bob Johnson", "customer_email": "bob@example.com", "items": ["itemC"]},
    "order789": {"customer_name": "Charlie Brown", "customer_email": "charlie@example.com", "items": ["itemA", "itemD"]},
}

def get_customer_details(order_id: str) -> tuple[str, str]:
    """
    Extracts customer name and email from an order ID.

    Args:
        order_id: The ID of the order.

    Returns:
        A tuple containing the customer's name and email.

    Raises:
        ValueError: If the order ID is not found.
    """
    if order_id in ORDERS:
        order = ORDERS[order_id]
        return order["customer_name"], order["customer_email"]
    else:
        raise ValueError(f"Order ID '{order_id}' not found.")

if __name__ == '__main__':
    # Example usage:
    try:
        name, email = get_customer_details("order123")
        print(f"Order order123: Customer Name: {name}, Email: {email}")

        name, email = get_customer_details("order456")
        print(f"Order order456: Customer Name: {name}, Email: {email}")

        # Example of an order not found
        name, email = get_customer_details("order000")
        print(f"Order order000: Customer Name: {name}, Email: {email}")
    except ValueError as e:
        print(e)
