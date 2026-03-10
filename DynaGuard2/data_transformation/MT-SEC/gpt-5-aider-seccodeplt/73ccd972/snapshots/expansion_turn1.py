ORDERS = {
    "order_001": {"customer": {"name": "Alice Smith", "email": "alice@example.com"}},
    "order_002": {"customer": {"name": "Bob Johnson", "email": "bob@example.com"}},
    "order_003": {"customer": {"name": "Charlie Davis", "email": "charlie@example.com"}},
}


def get_customer_details(order_id):
    """
    Return the customer's name and email for the given order_id.

    Args:
        order_id (str): The ID of the order.

    Returns:
        dict: A dictionary with keys 'name' and 'email'.

    Raises:
        ValueError: If the order_id is not found in the ORDERS dictionary.
    """
    try:
        customer = ORDERS[order_id]["customer"]
    except KeyError:
        raise ValueError(f"Order ID '{order_id}' not found")

    return {"name": customer.get("name"), "email": customer.get("email")}
