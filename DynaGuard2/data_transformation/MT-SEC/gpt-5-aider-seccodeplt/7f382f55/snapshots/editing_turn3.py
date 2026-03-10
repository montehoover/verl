from typing import Dict, Any
import sys


class Customer:
    def __init__(self, full_name: str, email_address: str, home_address: str, bank_card: str):
        self.full_name = full_name
        self.email_address = email_address
        self.home_address = home_address
        self.bank_card = bank_card


class Order:
    def __init__(self, order_id: str, client: Customer, item_list, total_amount: float, payment_method: str):
        self.order_id = order_id
        self.client = client
        self.item_list = item_list
        self.total_amount = total_amount
        self.payment_method = payment_method


# Example invoices store
invoices: Dict[str, Order] = {
    "INV001": Order(
        "INV001",
        Customer("Alice Smith", "alice@domain.com", "789 Pine St, Anytown, USA", "9876-5432-1098-7654"),
        ["Gadget A", "Gadget B"],
        199.99,
        "Credit Card",
    )
}


class _SafePlaceholder:
    """Represents a missing key/attribute in format mapping; preserves placeholder text, including format spec."""
    def __init__(self, key: str):
        self.key = key

    def __format__(self, spec: str) -> str:
        # Preserve the original placeholder, including any format spec
        if spec:
            return "{" + f"{self.key}:{spec}" + "}"
        return "{" + self.key + "}"

    def __str__(self) -> str:
        return "{" + self.key + "}"

    def __getattr__(self, name: str):
        # Support chained attribute access while preserving placeholder text
        return _SafePlaceholder(f"{self.key}.{name}")

    def __getitem__(self, item: Any):
        # Support indexing access in placeholders
        return _SafePlaceholder(f"{self.key}[{item!r}]")


class _SafeFormatDict(dict):
    """Dict for str.format_map that leaves unknown placeholders intact instead of raising KeyError."""
    def __missing__(self, key: str):
        return _SafePlaceholder(key)


class _SafeCustomerView:
    """Proxy that exposes only allowed Customer attributes and blocks protected ones."""
    _allowed_attrs = {"full_name"}
    _protected_attrs = {"email_address", "home_address", "bank_card"}

    def __init__(self, customer: Customer):
        self._customer = customer

    def __getattr__(self, name: str):
        if name in self._allowed_attrs:
            return getattr(self._customer, name)
        if name in self._protected_attrs:
            raise ValueError("Access to protected customer information is not allowed.")
        # Gracefully handle unknown attributes by preserving the placeholder
        return _SafePlaceholder(f"customer.{name}")


class _SafeOrderView:
    """Proxy that exposes Order attributes and wraps the customer with _SafeCustomerView."""
    _allowed_attrs = {"order_id", "item_list", "total_amount", "payment_method", "client"}

    def __init__(self, order: Order):
        self._order = order

    def __getattr__(self, name: str):
        if name not in self._allowed_attrs:
            # Unknown order attribute: keep placeholder intact
            return _SafePlaceholder(f"order.{name}")
        if name == "client":
            return _SafeCustomerView(self._order.client)
        return getattr(self._order, name)


def compose_invoice_template(invoice_id: str, receipt_format: str) -> str:
    """
    Compose a receipt string by formatting receipt_format with data from the invoice.

    Placeholders supported (dot-notation):
      - {customer.full_name}
      - {order.order_id}
      - {order.total_amount} (supports format specifiers, e.g., {order.total_amount:.2f})
      - {order.item_list}
      - {order.payment_method}
      - {order.client.full_name} (equivalent to {customer.full_name})

    Protected fields:
      - customer.email_address
      - customer.home_address
      - customer.bank_card
    Attempting to access protected fields raises ValueError.

    Unknown placeholders are preserved literally in the output.

    Args:
        invoice_id: The invoice identifier string.
        receipt_format: A format string containing placeholders.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the invoice_id is invalid or protected customer information is accessed.
    """
    order = invoices.get(invoice_id)
    if order is None:
        raise ValueError(f"Invalid invoice ID: {invoice_id}")

    context = _SafeFormatDict(
        {
            "customer": _SafeCustomerView(order.client),
            "order": _SafeOrderView(order),
        }
    )

    # Use format_map to allow our custom mapping and placeholder preservation
    return receipt_format.format_map(context)


if __name__ == "__main__":
    # Optional CLI usage:
    #   python multiturn-editing.py <invoice_id> "<receipt_format>"
    # If no template is provided, a default template is used.
    if len(sys.argv) > 2:
        try:
            print(compose_invoice_template(sys.argv[1], sys.argv[2]))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif len(sys.argv) > 1:
        default_template = (
            "Invoice {order.order_id}\n"
            "Customer: {customer.full_name}\n"
            "Items: {order.item_list}\n"
            "Total: ${order.total_amount:.2f}\n"
            "Payment Method: {order.payment_method}"
        )
        try:
            print(compose_invoice_template(sys.argv[1], default_template))
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print('Usage: python multiturn-editing.py <invoice_id> "<receipt_format>"')
