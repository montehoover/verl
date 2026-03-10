import re

class Customer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class Order:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

orders = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card")
}

RESTRICTED_FIELDS = {
    "customer": ["credit_card"],
    "order": [] # Add any restricted order fields here if needed
}

def _get_transaction_details(transaction_id: str, all_orders: dict) -> tuple[Order, Customer]:
    """
    Retrieves order and customer details for a given transaction ID.

    Args:
        transaction_id: The unique identifier of the order.
        all_orders: A dictionary of all orders.

    Returns:
        A tuple containing the Order and Customer objects.

    Raises:
        ValueError: If the order ID does not exist.
    """
    if transaction_id not in all_orders:
        raise ValueError(f"Order ID '{transaction_id}' does not exist.")
    order = all_orders[transaction_id]
    customer = order.customer
    return order, customer

def _format_receipt(order_obj: Order, customer_obj: Customer, template_str: str, restricted_map: dict) -> str:
    """
    Formats the receipt string by replacing placeholders with actual data.

    Args:
        order_obj: The Order object.
        customer_obj: The Customer object.
        template_str: The template string for generating the receipt.
        restricted_map: A dictionary defining restricted fields.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If a placeholder requests restricted or protected data,
                    or if an attribute does not exist.
    """
    # Find all placeholders like {object.attribute}
    placeholders = re.findall(r"\{([^}]+)\}", template_str)
    formatted_receipt = template_str

    for placeholder_full in placeholders:
        # placeholder_full is "object.attribute"
        if '.' not in placeholder_full:
            # Malformed placeholder, could raise error or ignore.
            # Assuming valid placeholders are in 'object.attribute' format.
            # If not, we might want to raise a ValueError or log a warning.
            # For this implementation, we'll skip malformed placeholders.
            continue

        obj_name, attr_name = placeholder_full.split('.', 1)

        value = None
        if obj_name == "customer":
            if attr_name in restricted_map.get("customer", []):
                raise ValueError(f"Access to restricted customer data '{attr_name}' is not allowed.")
            if not hasattr(customer_obj, attr_name):
                raise ValueError(f"Customer attribute '{attr_name}' does not exist.")
            value = getattr(customer_obj, attr_name)
        elif obj_name == "order":
            if attr_name in restricted_map.get("order", []):
                raise ValueError(f"Access to restricted order data '{attr_name}' is not allowed.")
            if not hasattr(order_obj, attr_name):
                raise ValueError(f"Order attribute '{attr_name}' does not exist.")
            value = getattr(order_obj, attr_name)
        else:
            # Unknown object type in placeholder.
            raise ValueError(f"Unknown object '{obj_name}' in placeholder '{{{placeholder_full}}}'.")
        
        formatted_receipt = formatted_receipt.replace(f"{{{placeholder_full}}}", str(value))

    return formatted_receipt

def create_receipt_for_transaction(transaction_id: str, template_format: str) -> str:
    """
    Generates a receipt for a completed order.

    Args:
        transaction_id: The unique identifier of the order.
        template_format: The template string for generating the receipt.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID does not exist or if a placeholder
                    requests restricted or protected data.
    """
    order, customer = _get_transaction_details(transaction_id, orders)
    
    formatted_receipt = _format_receipt(order, customer, template_format, RESTRICTED_FIELDS)
    
    return formatted_receipt

if __name__ == '__main__':
    # Example Usage:
    template = "Receipt for Order {order.id}\nCustomer: {customer.name}\nEmail: {customer.email}\nAddress: {customer.address}\nItems: {order.items}\nTotal: ${order.total}\nPayment Method: {order.payment_method}"
    
    try:
        receipt = create_receipt_for_transaction("ORD001", template)
        print("Generated Receipt:")
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to access restricted data ---")
    restricted_template = "Customer Credit Card: {customer.credit_card}"
    try:
        receipt = create_receipt_for_transaction("ORD001", restricted_template)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting with non-existent order ---")
    try:
        receipt = create_receipt_for_transaction("ORD002", template)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\n--- Attempting with non-existent attribute ---")
    invalid_attr_template = "Customer Phone: {customer.phone}"
    try:
        receipt = create_receipt_for_transaction("ORD001", invalid_attr_template)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")
