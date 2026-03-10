import re
import logging

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

# Define allowed attributes at module level for helper functions
ALLOWED_CUSTOMER_ATTRS = {"name", "email", "address"}
ALLOWED_ORDER_ATTRS = {"id", "items", "total", "payment_method"}

def _validate_and_get_placeholder_value(placeholder_key: str, customer: Customer, order: Order) -> str:
    """
    Validates a single placeholder key and retrieves/formats its value.

    Args:
        placeholder_key: The placeholder key (e.g., "customer.name").
        customer: The customer object.
        order: The order object.

    Returns:
        The string value for the placeholder.

    Raises:
        ValueError: If the placeholder is invalid, restricted, or refers to unknown attributes.
    """
    parts = placeholder_key.split('.', 1)
    if len(parts) != 2:
        # This case should ideally not be reached if re.findall is used correctly
        # and _replace_placeholders filters, but as a safeguard for direct calls:
        raise ValueError(f"Invalid placeholder format: '{placeholder_key}'")

    object_name, attr_name = parts[0], parts[1]
    value_to_insert = None

    if object_name == "customer":
        if attr_name == "credit_card":  # Explicitly restricted attribute
            raise ValueError("Access to restricted customer data (credit_card) is not allowed.")
        if attr_name in ALLOWED_CUSTOMER_ATTRS:
            value_to_insert = getattr(customer, attr_name)
        else:
            raise ValueError(f"Invalid or non-allowed placeholder: 'customer.{attr_name}'")
    elif object_name == "order":
        if attr_name in ALLOWED_ORDER_ATTRS:
            value_to_insert = getattr(order, attr_name)
            # Special formatting for certain attributes
            if attr_name == "items" and isinstance(value_to_insert, list):
                value_to_insert = ", ".join(map(str, value_to_insert))
            elif attr_name == "total" and isinstance(value_to_insert, (int, float)):
                value_to_insert = f"{value_to_insert:.2f}"
        else:
            raise ValueError(f"Invalid or non-allowed placeholder: 'order.{attr_name}'")
    else:
        raise ValueError(f"Invalid placeholder: Unknown object type '{object_name}' in '{placeholder_key}'")

    if value_to_insert is None:
        # This case should ideally be caught by earlier checks.
        raise ValueError(f"Could not resolve placeholder: '{placeholder_key}'")
        
    return str(value_to_insert)

def _replace_placeholders(template_string: str, customer: Customer, order: Order) -> str:
    """
    Replaces all valid placeholders in the template string with data from customer and order.
    """
    receipt = template_string
    # Find all placeholders like {key}
    placeholders_found = re.findall(r"\{([^}]+)\}", template_string)

    for placeholder_key in placeholders_found:
        # Validate format before attempting to get value
        if '.' not in placeholder_key: 
            # Simple keys like {foo} without object.attribute are not processed by _validate_and_get_placeholder_value
            # Depending on requirements, these could raise an error or be ignored.
            # For now, let's assume they are not dynamically replaced by this logic.
            continue

        value = _validate_and_get_placeholder_value(placeholder_key, customer, order)
        full_placeholder_to_replace = f"{{{placeholder_key}}}"
        receipt = receipt.replace(full_placeholder_to_replace, value)
    
    return receipt

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
    logger = logging.getLogger(__name__)
    # Initialize logger if it's not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()  # Outputs to console
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # To prevent messages from being passed to the root logger if it also has handlers
        # and to avoid duplicate messages if the root logger is configured.
        # For this specific setup, this ensures only our handler processes these logs.
        logger.propagate = False


    logger.info(f"Attempting to generate receipt for order ID: {order_identifier}.")
    logger.debug(f"Using template: \"{template_string}\"") # Using debug for potentially long template

    try:
        if order_identifier not in orders:
            raise ValueError(f"Order ID '{order_identifier}' does not exist.")

        order = orders[order_identifier]
        customer = order.customer

        receipt = _replace_placeholders(template_string, customer, order)
        
        logger.info(f"Successfully generated receipt for order ID: {order_identifier}.")
        logger.debug(f"Formatted receipt: \"{receipt}\"") # Using debug for potentially long receipt
        return receipt
    except ValueError as e:
        logger.error(f"Failed to generate receipt for order ID {order_identifier}: {e}")
        raise
