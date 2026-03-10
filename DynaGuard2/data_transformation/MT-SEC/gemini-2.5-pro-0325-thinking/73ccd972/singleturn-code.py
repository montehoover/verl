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

def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    """
    Generates a receipt for a completed order by substituting placeholders in a template.

    Args:
        purchase_id: The unique identifier of the order.
        receipt_template: The template string for generating the receipt.
                          Placeholders should be in the format {object.attribute},
                          e.g., {customer.name} or {order.total}.

    Returns:
        The formatted receipt string with placeholders replaced by order and customer details.

    Raises:
        ValueError: If the order ID does not exist, if a placeholder is malformed,
                    if a placeholder requests restricted or protected customer data,
                    or if a placeholder requests a non-existent attribute.
    """
    order = orders.get(purchase_id)
    if order is None:
        raise ValueError("Order ID does not exist")

    customer = order.customer
    
    # Define restricted customer fields. Accessing these via placeholders is forbidden.
    RESTRICTED_CUSTOMER_FIELDS = {"email", "address", "credit_card"}

    data_map = {}
    formatter = string.Formatter()

    for _, field_name, _, _ in formatter.parse(receipt_template):
        if field_name is None:
            continue # Skip literal text parts

        parts = field_name.split('.', 1) # Split only on the first dot, e.g., "customer.name"
        if len(parts) != 2:
            raise ValueError(f"Malformed placeholder: {{{field_name}}}")
        
        obj_name, attr_name = parts[0], parts[1]

        if obj_name == "customer":
            if attr_name in RESTRICTED_CUSTOMER_FIELDS:
                raise ValueError(
                    f"Placeholder {{{field_name}}} requests restricted or protected data from the customer."
                )
            try:
                data_map[field_name] = getattr(customer, attr_name)
            except AttributeError:
                raise ValueError(
                    f"Invalid attribute '{attr_name}' for customer in placeholder {{{field_name}}}"
                )
        elif obj_name == "order":
            try:
                # Accessing order.customer is allowed; its string representation will be used.
                data_map[field_name] = getattr(order, attr_name)
            except AttributeError:
                raise ValueError(
                    f"Invalid attribute '{attr_name}' for order in placeholder {{{field_name}}}"
                )
        else:
            raise ValueError(f"Invalid object type '{obj_name}' in placeholder {{{field_name}}}")
            
    return receipt_template.format_map(data_map)
