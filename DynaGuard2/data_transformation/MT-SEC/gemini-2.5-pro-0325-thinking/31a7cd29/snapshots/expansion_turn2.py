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

# Allowed placeholders
VALID_PLACEHOLDERS = {
    "customer.name",
    "order.id",
    "order.items",
    "order.total",
}

# Restricted placeholders (if any, for future use)
RESTRICTED_PLACEHOLDERS = {
    # "customer.address", # Example of a restricted placeholder
}

def check_template_placeholders(template_string: str) -> bool:
    """
    Verifies that all placeholders in a template string are valid.
    Placeholders should be in the format {object.attribute}.

    Args:
        template_string: The template string to check.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid or restricted.
    """
    import re
    placeholders = re.findall(r"\{(.+?)\}", template_string)

    for ph in placeholders:
        if ph in RESTRICTED_PLACEHOLDERS:
            raise ValueError(f"Placeholder '{{{ph}}}' is restricted.")
        if ph not in VALID_PLACEHOLDERS:
            raise ValueError(f"Placeholder '{{{ph}}}' is invalid.")
    return True

if __name__ == '__main__':
    # Example usage for get_order_by_id:
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

    print("\n--- Checking Template Placeholders ---")
    # Example usage for check_template_placeholders:
    valid_template_1 = "Receipt for {customer.name}. Order ID: {order.id}. Total: {order.total}."
    valid_template_2 = "Items: {order.items}."
    invalid_template_1 = "Dear {customer.name}, your order {order.number} is ready." # order.number is invalid
    invalid_template_2 = "Details: {order.id}, {customer.email}" # customer.email is invalid
    # restricted_template_1 = "Address: {customer.address}" # Assuming customer.address is restricted

    templates_to_test = {
        "Valid Template 1": valid_template_1,
        "Valid Template 2": valid_template_2,
        "Invalid Template (order.number)": invalid_template_1,
        "Invalid Template (customer.email)": invalid_template_2,
        # "Restricted Template (customer.address)": restricted_template_1,
    }

    for name, template in templates_to_test.items():
        try:
            check_template_placeholders(template)
            print(f"'{name}' is valid.")
        except ValueError as e:
            print(f"Error in '{name}': {e}")
