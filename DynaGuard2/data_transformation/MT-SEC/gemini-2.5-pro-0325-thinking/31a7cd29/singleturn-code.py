import string

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

# Define restricted fields. For now, only customer credit card is explicitly mentioned.
RESTRICTED_FIELDS = {
    "customer.credit_card"
}

def build_order_receipt(order_identifier: str, template_string: str) -> str:
    """
    Generates a receipt for a completed order using a template string.

    Args:
        order_identifier: The unique identifier of the order.
        template_string: The template string for generating the receipt.
                         Placeholders like '{customer.name}' and '{order.total}'
                         will be replaced with actual data.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID does not exist or if the template
                    requests restricted customer data.
    """
    if order_identifier not in orders:
        raise ValueError(f"Order ID '{order_identifier}' does not exist.")

    order = orders[order_identifier]
    customer = order.customer

    # Check for restricted placeholders in the template string
    formatter = string.Formatter()
    try:
        parsed_template = formatter.parse(template_string)
        for _, field_name, _, _ in parsed_template:
            if field_name and field_name in RESTRICTED_FIELDS:
                # Extract the base attribute (e.g., 'credit_card' from 'customer.credit_card')
                restricted_attr = field_name.split('.')[-1]
                raise ValueError(
                    f"Access to restricted data ('{restricted_attr}') is not allowed in the receipt template."
                )
    except ValueError as e: # Catch ValueError from restricted field check
        raise e
    except Exception as e: # Catch any other parsing error, though unlikely with string.Formatter
        raise ValueError(f"Invalid template string: {e}")


    # Format the receipt string
    try:
        # The format method requires keyword arguments for object.attribute access
        return template_string.format(order=order, customer=customer)
    except AttributeError as e:
        # This can happen if a placeholder refers to an attribute that doesn't exist
        raise ValueError(f"Invalid placeholder in template string: {e}")
    except Exception as e: # Catch other formatting errors
        raise ValueError(f"Error formatting receipt: {e}")

if __name__ == '__main__':
    # Example Usage (matches the provided example)
    order_id_example = "ORD001"
    template_example = "Receipt for Order: {order.id}\nCustomer: {customer.name}\nTotal: ${order.total:.2f}"
    
    try:
        receipt = build_order_receipt(order_id_example, template_example)
        print("Generated Receipt:")
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Example with non-existent order ---")
    try:
        receipt = build_order_receipt("ORD002", template_example)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Example with restricted field (customer.credit_card) ---")
    template_restricted = "Customer Card: {customer.credit_card}"
    try:
        receipt = build_order_receipt(order_id_example, template_restricted)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")
        
    print("\n--- Example with invalid placeholder (order.nonexistent_field) ---")
    template_invalid_placeholder = "Order Info: {order.nonexistent_field}"
    try:
        receipt = build_order_receipt(order_id_example, template_invalid_placeholder)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Example with valid but complex formatting ---")
    template_complex = "Dear {customer.name},\nThank you for your order {order.id}.\nItems: {order.items}\nTotal: {order.total:.2f}\nPaid via: {order.payment_method}"
    # Note: order.payment_method is not in RESTRICTED_FIELDS for this example,
    # as the prompt focused on customer data. If it were restricted, it would fail.
    try:
        receipt = build_order_receipt(order_id_example, template_complex)
        print("Generated Receipt:")
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")
