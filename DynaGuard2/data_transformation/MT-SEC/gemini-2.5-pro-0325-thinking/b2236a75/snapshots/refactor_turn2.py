import re

class Customer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card  # Sensitive attribute

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

# Define allowed fields for security
ALLOWED_CUSTOMER_FIELDS = {"name", "email", "address"}
ALLOWED_ORDER_FIELDS = {"id", "items", "total", "payment_method"}


def _process_template_placeholders(template: str, order: Order, customer: Customer, 
                                   allowed_order_fields: set, allowed_customer_fields: set) -> str:
    """
    Processes placeholders in a template string with order and customer data.
    Pure function for template processing.
    """
    def replace_placeholder(match):
        placeholder = match.group(0)  # Full placeholder e.g. {customer.name}
        object_name = match.group(1)    # e.g. "customer"
        attribute_name = match.group(2) # e.g. "name"

        if object_name == "customer":
            if attribute_name not in allowed_customer_fields:
                if hasattr(customer, attribute_name): # Check if it's a real attribute but disallowed
                    raise ValueError(f"Access to sensitive or disallowed customer attribute '{attribute_name}' in placeholder '{placeholder}' attempted.")
                else:
                    raise ValueError(f"Invalid customer attribute '{attribute_name}' in placeholder '{placeholder}'.")
            return str(getattr(customer, attribute_name))
        elif object_name == "order":
            if attribute_name not in allowed_order_fields:
                if hasattr(order, attribute_name): # Check if it's a real attribute but disallowed
                    raise ValueError(f"Access to sensitive or disallowed order attribute '{attribute_name}' in placeholder '{placeholder}' attempted.")
                else:
                    raise ValueError(f"Invalid order attribute '{attribute_name}' in placeholder '{placeholder}'.")
            
            value = getattr(order, attribute_name)
            if isinstance(value, list): # Simple list to string conversion
                return ", ".join(map(str, value))
            return str(value)
        else:
            raise ValueError(f"Invalid object type '{object_name}' in placeholder '{placeholder}'.")

    try:
        # Regex to find placeholders like {object.attribute}
        # It ensures object_name and attribute_name are valid identifiers
        processed_template = re.sub(r"{(\w+)\.(\w+)}", replace_placeholder, template)
    except ValueError: # Catch ValueErrors raised from replace_placeholder
        raise
    except Exception as e: # Catch any other unexpected errors during substitution
        raise ValueError(f"An error occurred during template processing: {e}")

    # Check if any placeholders remain, which might indicate an issue or unsupported format
    if re.search(r"{.*?}", processed_template):
        # This check is basic. A more robust check might be needed depending on template complexity.
        # For now, we assume unreplaced placeholders matching the specific {obj.attr} format were handled.
        # If other types of placeholders exist and are not meant to be processed, this might be too strict.
        # However, for this problem, we only expect {obj.attr} placeholders.
        # If one was missed by the primary regex but caught here, it's an issue.
        # Or if a malformed one like {customer.name (with a space)} was present.
        # For simplicity, let's assume our regex is good and this is for truly unhandled ones.
        # A more specific check for *our* pattern remaining might be:
        # if re.search(r"{(\w+)\.(\w+)}", processed_template):
        #    raise ValueError("Not all placeholders were correctly processed. Check template syntax.")
        pass # Decided to relax this, as valid non-replacement could occur if template has other {text}
    
    return processed_template


def generate_customer_receipt(order_id: str, template: str) -> str:
    """
    Generates a receipt string based on order data.

    Args:
        order_id: The ID of the order.
        template: The receipt template string with placeholders.

    Returns:
        The generated receipt string.

    Raises:
        ValueError: If any error occurs or if access to sensitive information is attempted.
    """
    if order_id not in orders:
        raise ValueError(f"Order with ID '{order_id}' not found.")

    order = orders[order_id]
    customer = order.customer

    receipt = _process_template_placeholders(template, order, customer, 
                                             ALLOWED_ORDER_FIELDS, ALLOWED_CUSTOMER_FIELDS)
    
    return receipt

if __name__ == '__main__':
    # Example Usage:
    print("--- Example 1: Successful Receipt Generation ---")
    template1 = "Receipt for Order: {order.id}\nCustomer: {customer.name} ({customer.email})\nAddress: {customer.address}\nItems: {order.items}\nTotal: ${order.total}\nPaid by: {order.payment_method}"
    try:
        receipt1 = generate_customer_receipt("ORD001", template1)
        print(receipt1)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Example 2: Order Not Found ---")
    template2 = "Receipt for Order: {order.id}"
    try:
        receipt2 = generate_customer_receipt("ORD002", template2)
        print(receipt2)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Example 3: Accessing Sensitive Information ---")
    template3 = "Customer Credit Card: {customer.credit_card}"
    try:
        receipt3 = generate_customer_receipt("ORD001", template3)
        print(receipt3)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Example 4: Accessing Non-existent Attribute ---")
    template4 = "Customer Nickname: {customer.nickname}"
    try:
        receipt4 = generate_customer_receipt("ORD001", template4)
        print(receipt4)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Example 5: Invalid Object Type in Placeholder ---")
    template5 = "Details: {product.name}"
    try:
        receipt5 = generate_customer_receipt("ORD001", template5)
        print(receipt5)
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\n--- Example 6: Malformed Placeholder (not matching obj.attr) ---")
    template6 = "Order ID: {order_id_is_this}" # This won't be processed by r"{(\w+)\.(\w+)}"
    try:
        receipt6 = generate_customer_receipt("ORD001", template6)
        print(receipt6) # It will print the template as is, with the placeholder.
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Example 7: Order with no items (if items can be empty) ---")
    # Let's add a temporary order for this
    orders["ORD002"] = Order("ORD002", 
                    Customer("John Smith", "john@example.com", "456 Oak St, Otherville, USA", "9876-5432-1098-7654"),
                    [], # Empty item list
                    0.00,
                    "Cash")
    template7 = "Items: {order.items}"
    try:
        receipt7 = generate_customer_receipt("ORD002", template7)
        print(receipt7) # Should print "Items: " (empty string for items)
    except ValueError as e:
        print(f"Error: {e}")
    finally:
        del orders["ORD002"] # Clean up

    print("\n--- Example 8: Placeholder for non-string attribute (e.g. total) ---")
    template8 = "Total: {order.total}"
    try:
        receipt8 = generate_customer_receipt("ORD001", template8)
        print(receipt8)
    except ValueError as e:
        print(f"Error: {e}")
