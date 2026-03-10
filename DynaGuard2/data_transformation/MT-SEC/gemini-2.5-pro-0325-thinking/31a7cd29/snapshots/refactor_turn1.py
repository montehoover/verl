import re

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

def build_order_receipt(order_identifier: str, template_string: str) -> str:
    """
    Generates a receipt for a completed order by substituting placeholders in a template.

    Args:
        order_identifier: The unique identifier of the order.
        template_string: The template string for generating the receipt.
                         Placeholders like '{customer.name}' and '{order.total}'
                         will be replaced.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID does not exist, or if a placeholder
                    requests restricted or protected data (e.g., customer.credit_card),
                    or if a placeholder is invalid or refers to unknown attributes/objects.
    """
    if order_identifier not in orders:
        raise ValueError(f"Order ID '{order_identifier}' does not exist.")

    order = orders[order_identifier]
    customer = order.customer

    # Define allowed attributes for safety
    allowed_customer_attrs = {"name", "email", "address"}
    allowed_order_attrs = {"id", "items", "total", "payment_method"}

    # Find all placeholders like {key} using re.findall to extract content between {}
    placeholders_found = re.findall(r"\{([^}]+)\}", template_string)
    
    receipt = template_string

    for placeholder_key in placeholders_found:
        full_placeholder_to_replace = f"{{{placeholder_key}}}"
        
        # Expecting placeholders of the form "object.attribute"
        parts = placeholder_key.split('.', 1) # Split only on the first dot
        
        if len(parts) != 2:
            # If placeholder_key is not in "object.attribute" format,
            # it's considered an unsupported placeholder format for dynamic replacement.
            # These will be ignored by this replacement logic and remain in the template.
            # If strict validation of all {} content is required, this part would need adjustment.
            continue

        object_name, attr_name = parts[0], parts[1]
        value_to_insert = None

        if object_name == "customer":
            if attr_name == "credit_card": # Explicitly restricted attribute
                raise ValueError("Access to restricted customer data (credit_card) is not allowed.")
            if attr_name in allowed_customer_attrs:
                value_to_insert = getattr(customer, attr_name)
            else:
                # Placeholder for an unknown or non-allowed customer attribute
                raise ValueError(f"Invalid or non-allowed placeholder: 'customer.{attr_name}'")
        elif object_name == "order":
            if attr_name in allowed_order_attrs:
                value_to_insert = getattr(order, attr_name)
                # Special formatting for certain attributes
                if attr_name == "items" and isinstance(value_to_insert, list):
                    value_to_insert = ", ".join(map(str, value_to_insert))
                elif attr_name == "total" and isinstance(value_to_insert, (int, float)):
                    value_to_insert = f"{value_to_insert:.2f}"
            else:
                # Placeholder for an unknown or non-allowed order attribute
                raise ValueError(f"Invalid or non-allowed placeholder: 'order.{attr_name}'")
        else:
            # Placeholder refers to an unknown object type
            raise ValueError(f"Invalid placeholder: Unknown object type '{object_name}' in '{placeholder_key}'")

        if value_to_insert is not None:
            # Replace all occurrences of this specific placeholder string
            receipt = receipt.replace(full_placeholder_to_replace, str(value_to_insert))
        # If value_to_insert is None, it implies an issue that should have been caught by
        # one of the ValueError conditions above (e.g. unknown attribute).

    return receipt
