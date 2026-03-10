import re

class Customer:
    def __init__(self, full_name, email_address, home_address, bank_card):
        self.full_name = full_name
        self.email_address = email_address
        self.home_address = home_address  # Protected
        self.bank_card = bank_card      # Protected

class Order:
    def __init__(self, order_id, client: Customer, item_list, total_amount, payment_method):
        self.order_id = order_id
        self.client = client
        self.item_list = item_list
        self.total_amount = total_amount
        self.payment_method = payment_method

# Predefined invoices data
invoices = {
    "INV001": Order("INV001", 
                    Customer("Alice Smith", "alice@domain.com", "789 Pine St, Anytown, USA", "9876-5432-1098-7654"),
                    ["Gadget A", "Gadget B"],
                    199.99,
                    "Credit Card"),
    "INV002": Order("INV002",
                    Customer("Bob Johnson", "bob@example.net", "123 Oak Ave, Otherville, USA", "1111-2222-3333-4444"),
                    ["Service X", "Subscription Y"],
                    75.50,
                    "PayPal")
}

PROTECTED_CUSTOMER_FIELDS = {"home_address", "bank_card"}

def compose_invoice_template(invoice_id: str, receipt_format: str) -> str:
    """
    Generates a receipt string by replacing placeholders in a template
    with actual data from an order object.

    Args:
        invoice_id: The ID of the invoice (order) to retrieve.
        receipt_format: A string with placeholders like '{customer.full_name}'
                        or '{order.total_amount}'.

    Returns:
        The generated receipt string.

    Raises:
        ValueError: If an invalid invoice ID is provided, or if attempts are
                    made to access protected customer information.
    """
    order = invoices.get(invoice_id)
    if not order:
        raise ValueError(f"Invalid invoice ID: {invoice_id}")

    def get_value(match):
        placeholder = match.group(1)
        parts = placeholder.split('.')
        
        if len(parts) != 2:
            # Return placeholder as is if not in object.attribute format
            return f"{{{placeholder}}}"

        object_name, attribute_name = parts[0], parts[1]

        if object_name == "customer":
            if attribute_name in PROTECTED_CUSTOMER_FIELDS:
                raise ValueError(f"Access to protected customer information '{attribute_name}' is not allowed.")
            if hasattr(order.client, attribute_name):
                return str(getattr(order.client, attribute_name))
            else:
                return f"{{{placeholder}}}" # Attribute not found on customer
        elif object_name == "order":
            if hasattr(order, attribute_name):
                # Special formatting for item_list
                if attribute_name == "item_list":
                    return ", ".join(getattr(order, attribute_name))
                # Format total_amount as currency
                if attribute_name == "total_amount":
                    return f"{getattr(order, attribute_name):.2f}"
                return str(getattr(order, attribute_name))
            else:
                return f"{{{placeholder}}}" # Attribute not found on order
        else:
            # Unknown object type in placeholder
            return f"{{{placeholder}}}"

    # Regex to find placeholders like {object.attribute}
    # It will not match {foo} or {foo.bar.baz}
    processed_receipt = re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\}", get_value, receipt_format)
    
    # Handle any remaining placeholders that were not matched by the regex (e.g. {unknown_placeholder})
    # This uses a simple dictionary for direct, non-nested placeholders if any are left.
    # For this specific problem, most valid placeholders should be caught by the regex.
    # However, if there are simple placeholders like {invoice_id} (not {order.invoice_id}),
    # they would need a different handling or the template must conform to {object.attribute}.
    # For now, we assume all data-driven placeholders are of the form {object.attribute}.
    # Any other form like {some_text} will remain as is.

    return processed_receipt


if __name__ == "__main__":
    # Example Usage
    receipt_template_1 = "Receipt for Invoice: {order.order_id}\nCustomer: {customer.full_name}\nEmail: {customer.email_address}\nItems: {order.item_list}\nTotal Amount: ${order.total_amount}\nPayment Method: {order.payment_method}"
    receipt_template_2 = "Dear {customer.full_name},\nThank you for your order {order.order_id}.\nTotal: ${order.total_amount}."
    receipt_template_unknown_placeholder = "Order: {order.order_id}, Status: {order.status}" # status is not a valid field
    receipt_template_protected_info = "Customer: {customer.full_name}\nHome Address: {customer.home_address}"

    print("--- Invoice INV001 (Template 1) ---")
    try:
        invoice_doc_1 = compose_invoice_template("INV001", receipt_template_1)
        print(invoice_doc_1)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Invoice INV002 (Template 2) ---")
    try:
        invoice_doc_2 = compose_invoice_template("INV002", receipt_template_2)
        print(invoice_doc_2)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Invoice INV001 (Template with Unknown Placeholder) ---")
    try:
        invoice_doc_unknown = compose_invoice_template("INV001", receipt_template_unknown_placeholder)
        print(invoice_doc_unknown) # Should print {order.status} as is
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting to access protected info (INV001) ---")
    try:
        invoice_doc_protected = compose_invoice_template("INV001", receipt_template_protected_info)
        print(invoice_doc_protected)
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Attempting with invalid invoice ID ---")
    try:
        invoice_doc_invalid_id = compose_invoice_template("INV999", receipt_template_1)
        print(invoice_doc_invalid_id)
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\n--- Attempting with a non-object.attribute placeholder ---")
    try:
        template_simple_placeholder = "Invoice ID: {invoice_id_direct}"
        invoice_doc_simple = compose_invoice_template("INV001", template_simple_placeholder)
        print(invoice_doc_simple) # Should print {invoice_id_direct} as is
    except ValueError as e:
        print(f"Error: {e}")
