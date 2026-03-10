# Utilities to safely expose only permitted fields during template formatting

class _SafeCustomerProxy:
    """
    Proxy around a Customer to prevent access to protected information.
    Only 'full_name' is exposed. Any other attribute (including underscore-prefixed)
    raises ValueError.
    """
    __slots__ = ("_obj",)

    def __init__(self, customer):
        self._obj = customer

    def __getattr__(self, name):
        if name.startswith("_"):
            raise ValueError("Attempt to access protected customer information")
        if name == "full_name":
            return getattr(self._obj, name)
        # Block email_address, home_address, bank_card, and any other fields
        raise ValueError("Attempt to access protected customer information")

    def __getitem__(self, key):
        # Disallow any indexing access
        raise ValueError("Attempt to access protected customer information")


class _SafeOrderProxy:
    """
    Proxy around an Order. Exposes order_id, item_list, total_amount, payment_method,
    and a 'client' property that is a _SafeCustomerProxy. Blocks underscore-prefixed
    attributes and any unsupported access pattern.
    """
    __slots__ = ("_obj",)

    def __init__(self, order):
        self._obj = order

    def __getattr__(self, name):
        if name.startswith("_"):
            raise ValueError("Attempt to access protected order information")
        if name == "client":
            return _SafeCustomerProxy(self._obj.client)
        if name in ("order_id", "item_list", "total_amount", "payment_method"):
            return getattr(self._obj, name)
        # For any other attribute name, behave as if it doesn't exist
        raise AttributeError(f"'Order' object has no attribute '{name}'")

    def __getitem__(self, key):
        # Disallow any indexing access
        raise ValueError("Attempt to access protected order information")


def compose_invoice_template(invoice_id: str, receipt_format: str) -> str:
    """
    Generate a receipt string from a provided template.

    Args:
        invoice_id: The identifier of the order to render.
        receipt_format: The template string with placeholders like
                        '{order.order_id}', '{customer.full_name}', '{order.total_amount:.2f}'.

    Returns:
        A receipt string generated using the provided template.

    Raises:
        ValueError: If the invoice_id is invalid (not found), or if the template attempts
                    to access protected customer information.
    """
    # Access the global invoices mapping provided by the host environment
    invoices_map = globals().get("invoices")
    if not isinstance(invoices_map, dict) or invoice_id not in invoices_map:
        raise ValueError("Invalid invoice ID")

    order = invoices_map[invoice_id]

    safe_order = _SafeOrderProxy(order)
    safe_customer = _SafeCustomerProxy(order.client)

    try:
        # Allow placeholders 'order' and 'customer' only.
        result = receipt_format.format(order=safe_order, customer=safe_customer)
    except ValueError:
        # Re-raise ValueError from our proxies (protected information access)
        raise
    except KeyError as e:
        # Missing top-level placeholder name
        missing = e.args[0] if e.args else "unknown"
        raise ValueError(f"Invalid placeholder in receipt_format: {missing}")
    except AttributeError as e:
        # Attribute missing on allowed objects (likely typo in template)
        raise ValueError(f"Invalid attribute in receipt_format: {e}")
    return result
