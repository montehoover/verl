from string import Formatter
from typing import List, Any


class User:
    def __init__(self, name: str, email: str, address: str, credit_card: str):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card


class Purchase:
    def __init__(self, id: str, client: User, items: List[str], total: float, payment_method: str):
        self.id = id
        self.client = client
        self.items = items
        self.total = total
        self.payment_method = payment_method


purchases = {
    "ORD002": Purchase(
        "ORD002",
        User("John Smith", "john.smith@example.com", "456 Park Ave, Metropolis, USA", "9876-5432-1098-7654"),
        ["Item A", "Item B"],
        150.50,
        "Credit Card",
    )
}


def _extract_field_paths(template: str) -> List[str]:
    """
    Return a list of field names (paths) used in the template. Format specs and conversions are ignored.
    Example: 'Receipt {order.id} {customer.name} {order.total:.2f}' -> ['order.id', 'customer.name', 'order.total']
    """
    paths: List[str] = []
    for literal, field_name, format_spec, conversion in Formatter().parse(template):
        if field_name:
            # Field name may include indexing or attributes; we keep as-is for validation.
            # format_spec is separate already (e.g., '.2f'), so field_name is clean.
            paths.append(field_name)
    return paths


def _validate_no_restricted_access(field_paths: List[str]) -> None:
    """
    Raise ValueError if the template attempts to access restricted customer data.
    Restricted attributes: credit_card, email, address.
    Blocks access via either {customer.<attr>} or {order.client.<attr>}.
    """
    restricted_attrs = {"credit_card", "email", "address"}

    def has_restricted_segment(path_parts: List[str]) -> bool:
        # If path begins with 'customer', check the next part against restricted attrs
        if len(path_parts) >= 2 and path_parts[0] == "customer" and path_parts[1] in restricted_attrs:
            return True
        # If path begins with 'order.client', check the next part
        if len(path_parts) >= 3 and path_parts[0] == "order" and path_parts[1] == "client" and path_parts[2] in restricted_attrs:
            return True
        return False

    for raw in field_paths:
        # Strip any indexing like [key] since we don't support it on our objects; still scan for obvious patterns
        # Split on '.' to analyze attribute chain
        parts: List[str] = []
        for segment in raw.split("."):
            # Normalize segments like 'client[foo]' -> 'client'
            clean = segment.split("[", 1)[0]
            parts.append(clean)
        if has_restricted_segment(parts):
            raise ValueError("Attempt to access restricted customer data in template")


def create_user_receipt(order_id: str, template: str) -> str:
    """
    Generate a receipt string by filling in placeholders in the template with order and customer data.

    Args:
        order_id: The unique identifier of the order.
        template: A format template string that may include placeholders such as:
                  {order.id}, {order.total}, {customer.name}, etc.
                  Standard format specs are supported, e.g., {order.total:.2f}.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order_id is invalid or if the template attempts to access restricted customer data.
    """
    # Validate order ID
    if order_id not in purchases:
        raise ValueError("Invalid order ID")

    order: Purchase = purchases[order_id]
    customer: User = order.client

    # Validate template for restricted access
    field_paths = _extract_field_paths(template)
    _validate_no_restricted_access(field_paths)

    # Perform formatting using attribute access
    try:
        result = template.format(order=order, customer=customer)
    except Exception as exc:
        # Re-raise as a ValueError to keep a clean API for the caller
        raise ValueError(f"Failed to format receipt: {exc}") from exc

    return result


# If you want to run a quick manual test, uncomment below:
# if __name__ == "__main__":
#     tpl = "Receipt for Order: {order.id}\nCustomer: {customer.name}\nTotal: ${order.total:.2f}"
#     print(create_user_receipt("ORD002", tpl))
