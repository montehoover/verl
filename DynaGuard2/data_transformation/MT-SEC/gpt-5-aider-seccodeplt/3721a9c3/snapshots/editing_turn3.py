from typing import Any, Dict
import string


class Buyer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card


class PurchaseOrder:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method


purchase_orders: Dict[str, PurchaseOrder] = {
    "ORD123": PurchaseOrder(
        "ORD123",
        Buyer("Alice Black", "alice@example.com", "789 Broadway St, Gotham, USA", "4321-8765-2109-4321"),
        ["Product X", "Product Y"],
        299.50,
        "Debit Card",
    )
}


class _MissingKey:
    def __init__(self, key: str):
        self.key = key

    def __format__(self, format_spec: str) -> str:
        # Preserve the original placeholder, including format spec, if any
        if format_spec:
            return f"{{{self.key}:{format_spec}}}"
        return f"{{{self.key}}}"

    def __str__(self) -> str:
        return f"{{{self.key}}}"

    def __getattr__(self, attr: str) -> "._MissingKey":
        # Support dotted chaining like {unknown.path.more}
        return _MissingKey(f"{self.key}.{attr}")

    def __getitem__(self, item: Any) -> "._MissingKey":
        # Support item access if ever used: {unknown[0]}
        return _MissingKey(f"{self.key}[{item!r}]")


class _SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            if key in kwargs:
                return kwargs[key]
            return _MissingKey(key)
        return super().get_value(key, args, kwargs)


class _OrderProxy:
    def __init__(self, po: PurchaseOrder):
        self._po = po
        self._allowed = {"id", "items", "total", "payment_method"}

    def __getattr__(self, attr: str):
        if attr in self._allowed:
            return getattr(self._po, attr, None)
        # Leave unknown placeholders intact
        return _MissingKey(f"order.{attr}")


class _CustomerProxy:
    def __init__(self, buyer: Buyer):
        self._buyer = buyer
        self._allowed = {"name"}
        self._restricted = {"email", "address", "credit_card"}

    def __getattr__(self, attr: str):
        if attr in self._allowed:
            return getattr(self._buyer, attr, None)
        if attr in self._restricted:
            raise ValueError(f"Attempt to access restricted customer data: {attr}")
        # Unknown, non-restricted fields are left intact
        return _MissingKey(f"customer.{attr}")


def generate_buyer_receipt(order_id: str, template: str) -> str:
    """
    Generate a formatted receipt string based on the given template.

    Placeholders supported (examples):
      - {order.id}
      - {order.total} or with specifiers {order.total:,.2f}
      - {order.payment_method}
      - {customer.name}

    Any attempt to access restricted customer data (email, address, credit_card)
    will raise a ValueError.

    Unknown placeholders are left intact in the output.
    """
    po = purchase_orders.get(order_id)
    if po is None:
        raise ValueError(f"Invalid order ID: {order_id}")

    formatter = _SafeFormatter()
    context = {
        "order": _OrderProxy(po),
        "customer": _CustomerProxy(po.customer),
    }
    return formatter.format(template, **context)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        print(generate_buyer_receipt(sys.argv[1], sys.argv[2]))
    elif len(sys.argv) == 2:
        default_template = (
            "Receipt for {order.id}\n"
            "Buyer: {customer.name}\n"
            "Items: {order.items}\n"
            "Total: ${order.total:,.2f}\n"
            "Payment: {order.payment_method}"
        )
        print(generate_buyer_receipt(sys.argv[1], default_template))
    else:
        print("Usage: python multiturn-editing.py <order_id> [template]")
