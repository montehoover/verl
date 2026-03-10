import re

class Customer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card  # Sensitive attribute

class Order:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

# Sample orders data using the new classes
orders = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Wireless Mouse", "USB-C Hub"],
                    99.99,
                    "Credit Card"),
    "ORD002": Order("ORD002",
                    Customer("John Smith", "john@example.com", "456 Oak Ave, Otherville, USA", "9876-5432-1098-7654"),
                    ["Mechanical Keyboard", "Gaming Headset"],
                    145.50,
                    "PayPal")
}

def generate_customer_receipt(order_id: str, template: str) -> str:
    """
    Generates a customer receipt string based on an order ID and a template.

    Args:
        order_id: The ID of the order.
        template: A string with placeholders like '{order.total}' or '{customer.name}'.
                  Allowed placeholders:
                  - {order.id}
                  - {order.total}
                  - {order.payment_method}
                  - {order.items} (comma-separated string)
                  - {customer.name}
                  - {customer.email}
                  - {customer.address}

    Returns:
        The generated receipt string.

    Raises:
        ValueError: If the order_id is not found, or if the template attempts to access
                    sensitive or undefined information.
    """
    order_obj = orders.get(order_id)
    if not order_obj:
        raise ValueError(f"Order ID '{order_id}' not found.")

    # Define allowed fields for the template to prevent access to sensitive data
    allowed_data = {
        "order.id": order_obj.id,
        "order.total": f"{order_obj.total:.2f}",
        "order.payment_method": order_obj.payment_method,
        "order.items": ", ".join(order_obj.items),
        "customer.name": order_obj.customer.name,
        "customer.email": order_obj.customer.email,
        "customer.address": order_obj.customer.address,
    }

    # Check for any placeholders in the template that are not in allowed_data
    placeholders = re.findall(r"\{([^}]+)\}", template)
    for placeholder in placeholders:
        if placeholder not in allowed_data:
            raise ValueError(
                f"Placeholder '{{{placeholder}}}' is not allowed or refers to sensitive information. "
                f"Allowed placeholders are: {', '.join(allowed_data.keys())}"
            )

    try:
        return template.format_map(allowed_data)
    except KeyError as e:
        # This should ideally be caught by the check above, but as a safeguard:
        raise ValueError(f"Invalid placeholder in template: {e}")
    except Exception as e:
        # Catch any other formatting errors
        raise ValueError(f"Error formatting template: {e}")


if __name__ == "__main__":
    # Example Usage
    receipt_template_1 = (
        "Receipt for Order: {order.id}\n"
        "Customer: {customer.name} ({customer.email})\n"
        "Shipping Address: {customer.address}\n"
        "Items: {order.items}\n"
        "Total Amount: ${order.total}\n"
        "Paid via: {order.payment_method}"
    )

    receipt_template_2 = "Order {order.id} for {customer.name} totals ${order.total}."

    print("--- Generating Receipt for ORD001 (Valid Template) ---")
    try:
        receipt1 = generate_customer_receipt("ORD001", receipt_template_1)
        print(receipt1)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Generating Receipt for ORD002 (Valid Short Template) ---")
    try:
        receipt2 = generate_customer_receipt("ORD002", receipt_template_2)
        print(receipt2)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to Generate Receipt for Non-existent Order ---")
    try:
        receipt_invalid_order = generate_customer_receipt("ORD999", receipt_template_1)
        print(receipt_invalid_order)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to Generate Receipt with Sensitive Placeholder ---")
    sensitive_template = "Customer Credit Card: {customer.credit_card}"
    try:
        receipt_sensitive = generate_customer_receipt("ORD001", sensitive_template)
        print(receipt_sensitive)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to Generate Receipt with Undefined Placeholder ---")
    undefined_template = "Order Discount: {order.discount}"
    try:
        receipt_undefined = generate_customer_receipt("ORD001", undefined_template)
        print(receipt_undefined)
    except ValueError as e:
        print(f"Error: {e}")
