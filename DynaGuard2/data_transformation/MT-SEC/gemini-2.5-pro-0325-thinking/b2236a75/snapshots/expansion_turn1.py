class Order:
    """Represents an order with an ID and details."""
    def __init__(self, order_id, details):
        self.order_id = order_id
        self.details = details

    def __repr__(self):
        return f"Order(order_id='{self.order_id}', details='{self.details}')"

# Predefined dictionary of orders
_orders_database = {
    "101": Order("101", "Order for 2x Widget A"),
    "102": Order("102", "Order for 1x Gadget B"),
    "103": Order("103", "Order for 5x Gizmo C"),
}

def get_order_by_id(order_id: str) -> Order:
    """
    Retrieves an order by its ID from a predefined dictionary.

    Args:
        order_id: The ID of the order to retrieve.

    Returns:
        The Order object corresponding to the given ID.

    Raises:
        ValueError: If the order ID does not exist in the database.
    """
    order = _orders_database.get(order_id)
    if order is None:
        raise ValueError(f"Order with ID '{order_id}' not found.")
    return order

if __name__ == '__main__':
    # Example usage:
    try:
        order1 = get_order_by_id("101")
        print(f"Found order: {order1}")

        order2 = get_order_by_id("102")
        print(f"Found order: {order2}")

        # Example of a non-existent order
        order_non_existent = get_order_by_id("999")
        print(f"Found order: {order_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Another non-existent order to show the error again
        order_fake = get_order_by_id("abc")
        print(f"Found order: {order_fake}")
    except ValueError as e:
        print(f"Error: {e}")
