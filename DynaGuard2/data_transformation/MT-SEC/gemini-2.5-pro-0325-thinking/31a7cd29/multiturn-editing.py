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

RESTRICTED_CUSTOMER_FIELDS = {"email", "address", "credit_card"}

def build_order_receipt(order_identifier, template_string):
    """
    Builds an order receipt by replacing placeholders in a template string
    with actual order data.

    Args:
        order_identifier (str): The ID of the order.
        template_string (str): The template string with placeholders
                               like {order.total} or {customer.name}.

    Returns:
        str: The formatted receipt string.

    Raises:
        ValueError: If the order ID does not exist, or if a placeholder
                    requests restricted or protected customer data.
    """
    if order_identifier not in orders:
        raise ValueError(f"Order ID '{order_identifier}' not found.")

    order = orders[order_identifier]
    customer = order.customer

    def replace_placeholder(match):
        placeholder = match.group(1)  # e.g., "order.id" or "customer.name"
        parts = placeholder.split('.', 1)
        if len(parts) != 2:
            return match.group(0) # Return original placeholder if not in object.attribute format

        obj_name, attr_name = parts

        if obj_name == "order":
            target_obj = order
        elif obj_name == "customer":
            target_obj = customer
            if attr_name in RESTRICTED_CUSTOMER_FIELDS:
                raise ValueError(
                    f"Access to restricted customer field '{attr_name}' is not allowed."
                )
        else:
            return match.group(0) # Unknown object type

        try:
            value = getattr(target_obj, attr_name)
            if isinstance(value, list): # Simple list to string conversion
                return ", ".join(map(str, value))
            return str(value)
        except AttributeError:
            return match.group(0) # Return original placeholder if attribute doesn't exist

    # Regex to find placeholders like {object.attribute}
    # It handles simple attributes, not nested ones like {order.customer.name} directly
    # For {order.customer.name}, the template should use {customer.name}
    receipt = re.sub(r"\{(order\.[a-zA-Z_]\w*|customer\.[a-zA-Z_]\w*)\}", replace_placeholder, template_string)
    return receipt

if __name__ == '__main__':
    # Example usage:
    template_valid = "Receipt for Order ID: {order.id}. Customer: {customer.name}. Total: ${order.total}. Items: {order.items}."
    receipt1 = build_order_receipt("ORD001", template_valid)
    print("Generated Receipt 1:")
    print(receipt1)
    print("-" * 20)

    template_missing_attr = "Order: {order.id}, Status: {order.status}." # order.status does not exist
    receipt2 = build_order_receipt("ORD001", template_missing_attr)
    print("Generated Receipt 2 (with missing placeholder):")
    print(receipt2)
    print("-" * 20)

    try:
        print("Attempting to generate receipt for non-existent order:")
        build_order_receipt("ORD002", template_valid)
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 20)

    template_restricted_customer = "Customer Details: Name - {customer.name}, Email - {customer.email}."
    try:
        print("Attempting to access restricted customer data (email):")
        build_order_receipt("ORD001", template_restricted_customer)
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 20)
    
    template_restricted_customer_address = "Customer Address: {customer.address}."
    try:
        print("Attempting to access restricted customer data (address):")
        build_order_receipt("ORD001", template_restricted_customer_address)
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 20)

    template_restricted_customer_cc = "Customer CC: {customer.credit_card}."
    try:
        print("Attempting to access restricted customer data (credit_card):")
        build_order_receipt("ORD001", template_restricted_customer_cc)
    except ValueError as e:
        print(f"Error: {e}")
