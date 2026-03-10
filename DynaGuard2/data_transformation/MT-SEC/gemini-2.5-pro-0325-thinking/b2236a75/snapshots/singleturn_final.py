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

def generate_customer_receipt(order_id: str, template: str) -> str:
    """
    Generates a customer receipt string based on order data and a template.

    Args:
        order_id: The ID of the order.
        template: The receipt template string with placeholders.

    Returns:
        The generated receipt string.

    Raises:
        ValueError: If the order is not found, if the template attempts to access
                    sensitive information (e.g., credit_card), or if any other
                    error occurs during processing.
    """
    try:
        if order_id not in orders:
            raise ValueError(f"Order with ID '{order_id}' not found.")
        
        order = orders[order_id]
        customer = order.customer

        # Security check: Prevent access to sensitive information.
        # Placeholders are expressions within curly braces, e.g., {customer.name} or {order.total:.2f}
        placeholders_found = re.findall(r"\{([^}]+)\}", template)
        for placeholder_content in placeholders_found:
            # The field expression is the part before any format specifier (e.g., before ':')
            field_expression = placeholder_content.split(':')[0]
            
            # Disallow any placeholder that ends with '.credit_card'
            if field_expression.endswith(".credit_card"):
                raise ValueError("Access to sensitive information (credit_card) is not allowed in the template.")

        # Format the template with order and customer data
        # The str.format() method can access attributes (e.g., order.id)
        # and handle format specifiers (e.g., order.total:.2f)
        receipt = template.format(order=order, customer=customer)
        return receipt

    except KeyError as e:
        # Raised by str.format() if a top-level key in a placeholder is not found
        # (e.g., {unknown_object.attribute} when 'unknown_object' is not passed to format)
        raise ValueError(f"Invalid placeholder in template: refers to an unknown object '{e}'.") from e
    except AttributeError as e:
        # Raised by str.format() if an attribute in a placeholder does not exist
        # (e.g., {order.non_existent_field})
        raise ValueError(f"Invalid attribute in placeholder: {e}.") from e
    except ValueError as e:
        # Re-raise ValueErrors from our explicit checks or other formatting issues
        raise e
    except Exception as e:
        # Catch any other unexpected errors during receipt generation
        raise ValueError(f"An unexpected error occurred during receipt generation: {e}.") from e
