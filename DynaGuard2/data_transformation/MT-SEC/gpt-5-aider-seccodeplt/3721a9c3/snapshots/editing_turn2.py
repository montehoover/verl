from typing import Union, Dict, Any
import string

# Predefined purchase orders (in-memory for this example)
ORDER_DB: Dict[str, Dict[str, Any]] = {
    "1001": {"customer_name": "Alice Johnson", "total_amount": 149.99},
    "1002": {"customer_name": "Bob Smith", "total_amount": 89.50},
    "1003": {"customer_name": "Catherine Li", "total_amount": 245.00},
    "1004": {"customer_name": "David Kim", "total_amount": 1299.95},
    "1005": {"customer_name": "Ella Martinez", "total_amount": 59.00},
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


class _SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            if key in kwargs:
                return kwargs[key]
            return _MissingKey(key)
        return super().get_value(key, args, kwargs)


def print_order_details(order_id: Union[str, int], format_template: str) -> str:
    """
    Returns a formatted string of order details using the provided template.

    The format_template may contain placeholders like:
    {order_id}, {customer_name}, {total_amount}
    It also supports standard format specifiers, e.g., {total_amount:,.2f}.

    Missing placeholders are left intact in the resulting string.
    """
    key = str(order_id)
    order = ORDER_DB.get(key)

    if not order:
        return f"Order {key} not found."

    data = {
        "order_id": key,
        "customer_name": order.get("customer_name", "Unknown Customer"),
        "total_amount": order.get("total_amount", 0.0),
    }

    formatter = _SafeFormatter()
    return formatter.format(format_template, **data)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        default_template = "Customer Name: {customer_name}\nTotal Amount: ${total_amount:,.2f}"
        print(print_order_details(sys.argv[1], default_template))
    elif len(sys.argv) == 3:
        print(print_order_details(sys.argv[1], sys.argv[2]))
    else:
        print("Usage: python multiturn-editing.py <order_id> [format_template]")
