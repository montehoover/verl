class Order:
    def __init__(self, order_id, customer_name, items, total_amount):
        self.order_id = order_id
        self.customer_name = customer_name
        self.items = items
        self.total_amount = total_amount

    def __repr__(self):
        return f"Order(order_id='{self.order_id}', customer_name='{self.customer_name}', items={self.items}, total_amount={self.total_amount})"

# Predefined dictionary of orders
_orders_database = {
    "101": Order("101", "Alice Smith", ["Laptop", "Mouse"], 1250.00),
    "102": Order("102", "Bob Johnson", ["Keyboard", "Monitor"], 375.50),
    "103": Order("103", "Carol Williams", ["Webcam", "Headset"], 120.75),
}

def get_order_by_id(order_id: str) -> Order:
    """
    Fetches an order by its ID from a predefined dictionary of orders.

    Args:
        order_id: The ID of the order to fetch.

    Returns:
        The Order object corresponding to the given order ID.

    Raises:
        ValueError: If the order ID is not found in the database.
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
        order_fake = get_order_by_id("ABC")
        print(f"Found order: {order_fake}")
    except ValueError as e:
        print(f"Error: {e}")
