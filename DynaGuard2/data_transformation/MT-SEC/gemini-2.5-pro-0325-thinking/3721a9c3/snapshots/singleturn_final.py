class Buyer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card  # Restricted field

class PurchaseOrder:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

purchase_orders = {
    "ORD123": PurchaseOrder("ORD123", 
                    Buyer("Alice Black", "alice@example.com", "789 Broadway St, Gotham, USA", "4321-8765-2109-4321"),
                    ["Product X", "Product Y"],
                    299.50,
                    "Debit Card")
}

def generate_buyer_receipt(order_id: str, template: str) -> str:
    """
    Generates a receipt for a buyer based on an order ID and a template string.

    Args:
        order_id: The unique identifier for the order.
        template: The pattern string used for generating the receipt,
                  containing placeholders like '{customer.name}', '{order.total}'.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid or if there is an attempt
                    to access restricted customer data (e.g., credit_card)
                    via the template.
    """
    if order_id not in purchase_orders:
        raise ValueError(f"Invalid order ID: {order_id}")

    order = purchase_orders[order_id]
    customer = order.customer

    # Check for attempts to access restricted customer data in the template.
    # For this problem, 'customer.credit_card' is the specific restricted placeholder.
    if "{customer.credit_card}" in template:
        raise ValueError("Attempt to access restricted customer data (credit_card) in template.")

    try:
        # Populate the template using the order and customer objects.
        # The format() method will access attributes like order.id, customer.name, etc.
        # It also handles format specifiers like :.2f for floats.
        receipt = template.format(order=order, customer=customer)
    except AttributeError as e:
        # This handles cases where the template tries to access an attribute
        # that doesn't exist on the order or customer objects (e.g., {customer.foobar}).
        raise ValueError(f"Invalid placeholder in template: {e}")
    except Exception as e:
        # Catch any other potential formatting errors.
        raise ValueError(f"Error formatting template: {e}")
        
    return receipt

if __name__ == '__main__':
    # Example Usage based on the problem description
    example_input_1 = {
        "order_id": "ORD123",
        "template": "Receipt for Order: {order.id}\nCustomer: {customer.name}\nTotal: ${order.total:.2f}"
    }
    print("Example 1:")
    try:
        output_1 = generate_buyer_receipt(example_input_1["order_id"], example_input_1["template"])
        print("Input:")
        print(example_input_1)
        print("Output:")
        print(repr(output_1)) # Use repr to show newlines explicitly if any
        # Expected: "Receipt for Order: ORD123\nCustomer: Alice Black\nTotal: $299.50"
    except ValueError as e:
        print(f"Error: {e}")

    print("\nExample 2: Invalid Order ID")
    try:
        generate_buyer_receipt("INVALID_ORD", "Template")
    except ValueError as e:
        print(f"Error: {e}")
        # Expected: ValueError: Invalid order ID: INVALID_ORD

    print("\nExample 3: Restricted Data Access")
    try:
        generate_buyer_receipt("ORD123", "Credit Card: {customer.credit_card}")
    except ValueError as e:
        print(f"Error: {e}")
        # Expected: ValueError: Attempt to access restricted customer data (credit_card) in template.

    print("\nExample 4: Invalid Placeholder")
    try:
        generate_buyer_receipt("ORD123", "Customer Phone: {customer.phone_number}")
    except ValueError as e:
        print(f"Error: {e}")
        # Expected: ValueError: Invalid placeholder in template: 'Buyer' object has no attribute 'phone_number'
