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

def create_receipt_for_transaction(transaction_id: str) -> str:
    """
    Verify an order by its transaction ID and return a confirmation message.
    Raises ValueError if the transaction ID is not found.
    """
    if not isinstance(transaction_id, str) or not transaction_id.strip():
        raise ValueError("transaction_id must be a non-empty string")

    if transaction_id in orders:
        return f"Transaction {transaction_id} exists in our system. Order confirmed."
    else:
        raise ValueError(f"Transaction ID {transaction_id} not found in our system.")
