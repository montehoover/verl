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

# Predefined orders data using the new classes
orders = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card")
}

RESTRICTED_FIELDS = {
    "customer.credit_card"
}

class ReceiptFormatter(string.Formatter):
    def __init__(self, order_obj: Order):
        super().__init__()
        self.order_obj = order_obj

    def get_value(self, key, args, kwargs):
        # key is the field name from the placeholder, e.g., "customer.name"
        
        if key in RESTRICTED_FIELDS:
            raise ValueError(f"Access to restricted field '{key}' is not allowed.")

        parts = key.split('.', 1) # Split only on the first dot
        if len(parts) == 2:
            obj_name, attr_name = parts
            
            target_obj = None
            if obj_name == "customer":
                target_obj = self.order_obj.customer
            elif obj_name == "order":
                target_obj = self.order_obj
            else:
                raise ValueError(f"Invalid object '{obj_name}' in placeholder '{{{key}}}'. Expected 'customer' or 'order'.")

            if hasattr(target_obj, attr_name):
                value = getattr(target_obj, attr_name)
                # Special formatting for list of items
                if obj_name == "order" and attr_name == "items" and isinstance(value, list):
                    return ", ".join(value)
                return value
            else:
                raise ValueError(f"Invalid attribute '{attr_name}' for object '{obj_name}' in placeholder '{{{key}}}'.")
        else:
            raise ValueError(f"Invalid placeholder format '{{{key}}}'. Expected 'object.attribute' (e.g., 'customer.name').")

def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    """
    Creates a formatted order receipt string from a template.

    Args:
        purchase_id: The ID of the order.
        receipt_template: A string template with placeholders like 
                          '{customer.name}', '{order.total}'.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID does not exist, a placeholder is invalid,
                    or a placeholder requests restricted data.
    """
    order = orders.get(purchase_id)
    if not order:
        raise ValueError(f"Order ID '{purchase_id}' not found.")

    formatter = ReceiptFormatter(order)
    
    try:
        return formatter.format(receipt_template)
    except KeyError as e: # Should ideally be caught by get_value logic, but as a fallback.
        raise ValueError(f"Invalid or unhandled placeholder in template: {{{e}}}")


if __name__ == '__main__':
    template1 = (
        "Receipt for Order: {order.id}\n"
        "Customer: {customer.name} ({customer.email})\n"
        "Address: {customer.address}\n"
        "Items: {order.items}\n"
        "Total Amount: ${order.total:.2f}\n"
        "Payment Method: {order.payment_method}"
    )

    template_restricted = "Customer Credit Card: {customer.credit_card}"
    template_invalid_attr = "Customer Phone: {customer.phone}" # 'phone' is not an attribute of Customer
    template_invalid_obj = "Details: {product.name}" # 'product' is not a valid object prefix
    template_malformed_placeholder = "Order ID: {orderid}" # Missing dot notation

    print("--- Generating Receipt for ORD001 ---")
    try:
        receipt1 = create_order_receipt("ORD001", template1)
        print(receipt1)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to generate receipt for non-existent order ORD999 ---")
    try:
        receipt_non_existent = create_order_receipt("ORD999", template1)
        print(receipt_non_existent)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to access restricted field customer.credit_card ---")
    try:
        receipt_restricted = create_order_receipt("ORD001", template_restricted)
        print(receipt_restricted)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to access invalid attribute customer.phone ---")
    try:
        receipt_invalid_attr = create_order_receipt("ORD001", template_invalid_attr)
        print(receipt_invalid_attr)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to access invalid object in placeholder product.name ---")
    try:
        receipt_invalid_obj = create_order_receipt("ORD001", template_invalid_obj)
        print(receipt_invalid_obj)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to use malformed placeholder {orderid} ---")
    try:
        receipt_malformed = create_order_receipt("ORD001", template_malformed_placeholder)
        print(receipt_malformed)
    except ValueError as e:
        print(f"Error: {e}")
