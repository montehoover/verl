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

def create_receipt_for_transaction(transaction_id: str, format_type: str) -> str:
    """
    Verify an order by its transaction ID and return a formatted receipt message.
    format_type can be "summary" or "detailed".
    Raises ValueError if the transaction ID is not found or if format_type is invalid.
    """
    if not isinstance(transaction_id, str) or not transaction_id.strip():
        raise ValueError("transaction_id must be a non-empty string")
    if not isinstance(format_type, str) or not format_type.strip():
        raise ValueError("format_type must be a non-empty string")

    fmt = format_type.strip().lower()
    if fmt not in ("summary", "detailed"):
        raise ValueError("format_type must be 'summary' or 'detailed'")

    order = orders.get(transaction_id)
    if order is None:
        raise ValueError(f"Transaction ID {transaction_id} not found in our system.")

    if fmt == "summary":
        return f"Order ID: {order.id}\nCustomer: {order.customer.name}"

    # detailed
    items_formatted = "\n".join(f" - {item}" for item in order.items)
    total_formatted = f"${order.total:.2f}"
    return (
        f"Order ID: {order.id}\n"
        f"Customer: {order.customer.name}\n"
        f"Items:\n{items_formatted}\n"
        f"Total: {total_formatted}\n"
        f"Payment Method: {order.payment_method}"
    )
