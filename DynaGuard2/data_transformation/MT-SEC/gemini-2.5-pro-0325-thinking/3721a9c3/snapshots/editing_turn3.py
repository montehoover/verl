import re

class Buyer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card  # Sensitive data

class PurchaseOrder:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer # This will be a Buyer object
        self.items = items # List of strings
        self.total = total # Float
        self.payment_method = payment_method

# Sample predefined purchase orders using the new classes
purchase_orders = {
    "ORD123": PurchaseOrder(
        id="ORD123", 
        customer=Buyer(
            name="Alice Black", 
            email="alice@example.com", 
            address="789 Broadway St, Gotham, USA", 
            credit_card="4321-8765-2109-4321" # Sensitive
        ),
        items=["Product X", "Product Y"],
        total=299.50,
        payment_method="Debit Card"
    ),
    "ORD456": PurchaseOrder(
        id="ORD456",
        customer=Buyer(
            name="Bob White",
            email="bob@example.com",
            address="123 Main St, Metropolis, USA",
            credit_card="1111-2222-3333-4444" # Sensitive
        ),
        items=["Product Z"],
        total=99.99,
        payment_method="Credit Card"
    )
}

def generate_buyer_receipt(order_id: str, template: str) -> str:
    """
    Generates a buyer receipt by formatting a template string with order details.

    Args:
        order_id: The ID of the purchase order.
        template: A string template with placeholders like {order.total} or {customer.name}.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is invalid, a placeholder is malformed,
                    or an attempt is made to access restricted customer data.
    """
    order = purchase_orders.get(order_id)
    if not order:
        raise ValueError(f"Invalid order ID: {order_id}")

    # Define allowed attributes for security and correctness
    allowed_order_attrs = {
        "id": lambda o: o.id,
        "items": lambda o: ", ".join(o.items),
        "total": lambda o: f"{o.total:.2f}", # Format total to 2 decimal places
        "payment_method": lambda o: o.payment_method
    }
    allowed_customer_attrs = {
        "name": lambda c: c.name,
        "email": lambda c: c.email,
        "address": lambda c: c.address
        # "credit_card" is intentionally omitted as it's restricted
    }
    
    restricted_customer_attrs = {"credit_card"}

    def replacer(match):
        placeholder = match.group(1)  # e.g., "order.total" or "customer.name"
        parts = placeholder.split('.', 1)

        if len(parts) != 2:
            raise ValueError(f"Malformed placeholder: {{{placeholder}}}. Expected format 'object.attribute'.")

        obj_name, attr_name = parts[0], parts[1]

        if obj_name == "order":
            if attr_name in allowed_order_attrs:
                return str(allowed_order_attrs[attr_name](order))
            else:
                raise ValueError(f"Invalid or restricted order attribute: {attr_name} in {{{placeholder}}}")
        elif obj_name == "customer":
            if attr_name in restricted_customer_attrs:
                raise ValueError(f"Access to restricted customer data attempted: {attr_name} in {{{placeholder}}}")
            if attr_name in allowed_customer_attrs:
                return str(allowed_customer_attrs[attr_name](order.customer))
            else:
                raise ValueError(f"Invalid or restricted customer attribute: {attr_name} in {{{placeholder}}}")
        else:
            raise ValueError(f"Unknown object type in placeholder: {obj_name} in {{{placeholder}}}")

    try:
        # Using re.sub with a replacer function to handle complex logic per placeholder
        return re.sub(r"\{([^}]+)\}", replacer, template)
    except AttributeError as e:
        # This might catch issues if an allowed attribute is unexpectedly missing from an object
        raise ValueError(f"Error accessing attribute for placeholder: {e}")


if __name__ == '__main__':
    # Example Usage
    receipt_template_v1 = (
        "Receipt for Order: {order.id}\n"
        "Customer: {customer.name} ({customer.email})\n"
        "Shipping Address: {customer.address}\n"
        "Items: {order.items}\n"
        "Total Amount: ${order.total}\n"
        "Paid via: {order.payment_method}\n"
        "Thank you for your purchase!"
    )

    receipt_template_v2 = (
        "Order Confirmation: {order.id}\n"
        "Buyer: {customer.name}\n"
        "Total: ${order.total}"
    )

    print("--- Generating Receipt for ORD123 (Valid) ---")
    try:
        receipt1 = generate_buyer_receipt("ORD123", receipt_template_v1)
        print(receipt1)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Generating Receipt for ORD456 (Valid) ---")
    try:
        receipt2 = generate_buyer_receipt("ORD456", receipt_template_v2)
        print(receipt2)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to Generate Receipt for Non-Existent Order ---")
    try:
        generate_buyer_receipt("ORD789", receipt_template_v1)
    except ValueError as e:
        print(f"Error: {e}") # Expected: Invalid order ID

    print("\n--- Attempting to Access Restricted Customer Data (credit_card) ---")
    restricted_template = "Customer Credit Card: {customer.credit_card}"
    try:
        generate_buyer_receipt("ORD123", restricted_template)
    except ValueError as e:
        print(f"Error: {e}") # Expected: Access to restricted customer data

    print("\n--- Attempting to Access Invalid Order Attribute ---")
    invalid_order_attr_template = "Order Status: {order.status}" # 'status' is not defined/allowed
    try:
        generate_buyer_receipt("ORD123", invalid_order_attr_template)
    except ValueError as e:
        print(f"Error: {e}") # Expected: Invalid or restricted order attribute

    print("\n--- Attempting to Access Invalid Customer Attribute ---")
    invalid_customer_attr_template = "Customer Phone: {customer.phone}" # 'phone' is not defined/allowed
    try:
        generate_buyer_receipt("ORD123", invalid_customer_attr_template)
    except ValueError as e:
        print(f"Error: {e}") # Expected: Invalid or restricted customer attribute

    print("\n--- Attempting with Malformed Placeholder ---")
    malformed_template = "Order Info: {order_id_only}" # Missing object prefix
    try:
        generate_buyer_receipt("ORD123", malformed_template)
    except ValueError as e:
        print(f"Error: {e}") # Expected: Malformed placeholder
