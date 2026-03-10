import string

ORDERS = {
    "1001": {"customer_name": "Alice Smith", "total_amount": 149.99},
    "1002": {"customer_name": "Bob Johnson", "total_amount": 89.50},
    "1003": {"customer_name": "Carol Lee", "total_amount": 230.00},
}


class _Placeholder:
    def __init__(self, key: str):
        self.key = key

    def __format__(self, format_spec: str) -> str:
        # Leave unknown placeholders unchanged, regardless of format spec
        return "{" + self.key + "}"

    def __str__(self) -> str:
        return "{" + self.key + "}"


class _SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, _Placeholder(key))
        return super().get_value(key, args, kwargs)


def print_order_details(order_id, template=None):
    """
    Retrieve and format basic order details (customer name and total amount)
    for the given order_id.

    Args:
        order_id (str|int): The identifier of the order to look up.
        template (str|None): A template string with placeholders such as
            '{order_id}', '{customer_name}', and '{total_amount}'. Unknown
            placeholders are left unchanged. If None, a default template
            is used.

    Returns:
        str: The formatted string with order details, or a not-found message.
    """
    oid = str(order_id)
    order = ORDERS.get(oid)
    if not order:
        return f"Order {oid} not found."

    customer_name = order.get("customer_name", "Unknown Customer")
    total_amount = order.get("total_amount", 0.0)

    context = {
        "order_id": oid,
        "customer_name": customer_name,
        "total_amount": total_amount,
    }

    if template is None:
        template = "Order {order_id} - Customer: {customer_name}\nTotal Amount: ${total_amount:,.2f}"

    formatter = _SafeFormatter()
    return formatter.format(template, **context)
