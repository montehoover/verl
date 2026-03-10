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

# Allowed attributes for placeholders to prevent arbitrary code execution via getattr
ALLOWED_CUSTOMER_ATTRIBUTES = {"name", "email", "address"}
ALLOWED_ORDER_ATTRIBUTES = {"id", "items", "total", "payment_method"}
RESTRICTED_CUSTOMER_ATTRIBUTES = {"credit_card"}

def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    """
    Generates a receipt for a completed order by substituting placeholders in a template.

    Args:
        purchase_id: The unique identifier of the order.
        receipt_template: The template string for generating the receipt.
                          Placeholders are in the format {object.attribute},
                          e.g., {customer.name} or {order.total}.

    Returns:
        The formatted receipt string with placeholders replaced by order and customer details.

    Raises:
        ValueError: If the order ID does not exist, or if a placeholder requests
                    restricted or protected data from the customer.
    """
    if purchase_id not in orders:
        raise ValueError(f"Order ID '{purchase_id}' does not exist.")

    order_obj = orders[purchase_id]
    customer_obj = order_obj.customer

    def replace_placeholder(match):
        object_name = match.group(1)
        attribute_name = match.group(2)

        if object_name == "customer":
            if attribute_name in RESTRICTED_CUSTOMER_ATTRIBUTES:
                raise ValueError(f"Access to restricted customer data ({attribute_name}) is not allowed.")
            if attribute_name not in ALLOWED_CUSTOMER_ATTRIBUTES:
                raise ValueError(f"Invalid or disallowed customer attribute: {attribute_name}")
            value = getattr(customer_obj, attribute_name)
        elif object_name == "order":
            if attribute_name not in ALLOWED_ORDER_ATTRIBUTES:
                raise ValueError(f"Invalid or disallowed order attribute: {attribute_name}")
            value = getattr(order_obj, attribute_name)
        else:
            # If the placeholder format is correct but object_name is neither 'customer' nor 'order'
            raise ValueError(f"Invalid object type in placeholder: {object_name}")
        
        return str(value)

    # Regex to find placeholders like {customer.name} or {order.total}
    # It ensures that object_name and attribute_name are valid identifiers
    placeholder_regex = r"{([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)}"
    
    try:
        formatted_receipt = re.sub(placeholder_regex, replace_placeholder, receipt_template)
    except ValueError: # Catch ValueErrors raised from replace_placeholder
        raise
    except Exception as e:
        # Catch other potential errors during getattr or regex processing
        raise ValueError(f"Error processing receipt template: {e}")

    # Check if any placeholders remain, which might indicate an invalid format not caught by regex
    # or an attribute that was not processed (e.g. due to a typo not matching allowed lists but also not raising error)
    # This check is more of a safeguard. The primary validation is within replace_placeholder.
    if re.search(r"{.*}", formatted_receipt): # A simple check for any remaining curly braces
        # This could be more sophisticated to only flag {object.attribute} style placeholders
        # For now, we assume if replace_placeholder didn't handle it and it's still there, it's an issue.
        # However, the current logic in replace_placeholder should catch all validly formatted {obj.attr}
        # and raise errors for disallowed/invalid ones.
        # This part might be redundant if all paths in replace_placeholder correctly raise or replace.
        pass # Or raise a more specific error if unreplaced placeholders are strictly forbidden.

    return formatted_receipt
