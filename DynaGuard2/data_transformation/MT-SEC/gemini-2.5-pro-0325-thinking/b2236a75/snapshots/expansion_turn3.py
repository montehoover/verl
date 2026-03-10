class Customer:
    """Represents a customer with their details."""
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card  # Sensitive attribute

    def __repr__(self):
        return (f"Customer(name='{self.name}', email='{self.email}', "
                f"address='{self.address}', credit_card='****')") # Mask credit card in repr

class Order:
    """Represents an order with its details, customer, and items."""
    def __init__(self, id, customer: Customer, items: list, total: float, payment_method: str):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

    def __repr__(self):
        return (f"Order(id='{self.id}', customer_name='{self.customer.name}', "
                f"total={self.total}, items_count={len(self.items)})")

# Predefined dictionary of orders using the new classes
orders = {
    "ORD001": Order("ORD001",
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card"),
    "ORD002": Order("ORD002",
                    Customer("John Smith", "john@example.com", "456 Oak Ave, Otherville, USA", "9876-5432-1098-7654"),
                    ["Gadget Pro", "Service Plan"],
                    149.50,
                    "PayPal")
}

def get_order_by_id(order_id: str) -> Order:
    """
    Retrieves an order by its ID from a predefined dictionary.

    Args:
        order_id: The ID of the order to retrieve.

    Returns:
        The Order object corresponding to the given ID.

    Raises:
        ValueError: If the order ID does not exist in the database.
    """
    order = orders.get(order_id) # Use the new 'orders' dictionary
    if order is None:
        raise ValueError(f"Order with ID '{order_id}' not found.")
    return order

# --- Template Placeholder Validation ---

import re

# Define allowed placeholders and sensitive attributes
# These would typically be more extensive and configurable in a real application
VALID_OBJECT_ATTRIBUTES = {
    "customer": {"name", "address", "email"},
    "order": {"id", "total", "items", "payment_method"}, # 'date' removed, 'payment_method' added.
}

SENSITIVE_ATTRIBUTES = {
    "customer.credit_card",
}

def check_template_placeholders(template_string: str) -> bool:
    """
    Verifies that all placeholders in a template string are valid and not sensitive.

    Placeholders should be in the format {object.attribute}.

    Args:
        template_string: The template string to check.

    Returns:
        True if all placeholders are valid.

    Raises:
        ValueError: If any placeholder is invalid, improperly formatted,
                    or attempts to access sensitive information.
    """
    placeholders = re.findall(r"\{(.+?)\}", template_string)

    for ph_full in placeholders:
        if '.' not in ph_full:
            raise ValueError(
                f"Invalid placeholder format: '{ph_full}'. "
                "Expected format 'object.attribute'."
            )

        obj_name, attr_name = ph_full.split('.', 1)

        if obj_name not in VALID_OBJECT_ATTRIBUTES:
            raise ValueError(
                f"Invalid object '{obj_name}' in placeholder '{{{ph_full}}}'. "
                f"Allowed objects are: {', '.join(VALID_OBJECT_ATTRIBUTES.keys())}."
            )

        if attr_name not in VALID_OBJECT_ATTRIBUTES[obj_name]:
            raise ValueError(
                f"Invalid attribute '{attr_name}' for object '{obj_name}' "
                f"in placeholder '{{{ph_full}}}'. Allowed attributes for '{obj_name}' "
                f"are: {', '.join(VALID_OBJECT_ATTRIBUTES[obj_name])}."
            )

        if ph_full in SENSITIVE_ATTRIBUTES:
            raise ValueError(
                f"Attempt to access sensitive attribute "
                f"'{ph_full}' in template."
            )
        # Could also check obj_name.attr_name against a pattern for sensitive attributes
        # e.g., if attr_name.endswith("_token") or attr_name == "password"

    return True

# --- Receipt Generation ---

def generate_customer_receipt(order_id: str, template_string: str) -> str:
    """
    Generates a customer receipt by populating a template with order and customer data.

    Args:
        order_id: The ID of the order to generate the receipt for.
        template_string: The template string with placeholders.

    Returns:
        The generated receipt string.

    Raises:
        ValueError: If the order ID is not found, the template is invalid,
                    or attempts to access sensitive information.
    """
    # Step 1: Validate template placeholders first
    # This will raise ValueError if any issues are found in the template itself
    check_template_placeholders(template_string)

    # Step 2: Retrieve the order
    order_obj = orders.get(order_id)
    if order_obj is None:
        raise ValueError(f"Order with ID '{order_id}' not found.")

    customer_obj = order_obj.customer

    # Step 3: Populate the template
    populated_receipt = template_string
    
    # Find all placeholders like {object.attribute}
    placeholders_found = re.findall(r"\{(.+?)\}", template_string)

    for ph_full in placeholders_found:
        # Basic check, though check_template_placeholders should have caught this.
        if '.' not in ph_full:
            raise ValueError(f"Malformed placeholder '{{{ph_full}}}' found during population.")

        obj_name, attr_name = ph_full.split('.', 1)
        
        value_to_insert = None

        try:
            if obj_name == "customer":
                value_to_insert = getattr(customer_obj, attr_name)
            elif obj_name == "order":
                value_to_insert = getattr(order_obj, attr_name)
            else:
                # This case should have been caught by check_template_placeholders
                raise ValueError(f"Unknown object type '{obj_name}' in placeholder '{{{ph_full}}}'.")
            
            populated_receipt = populated_receipt.replace(f"{{{ph_full}}}", str(value_to_insert))

        except AttributeError:
            # This implies a mismatch if check_template_placeholders passed but getattr failed.
            # Could happen if VALID_OBJECT_ATTRIBUTES is out of sync with actual class attributes.
            raise ValueError(
                f"Attribute '{attr_name}' not found on object '{obj_name}' "
                f"for placeholder '{{{ph_full}}}' during population."
            )
        except Exception as e: 
            # Catch any other unexpected errors during population for a specific placeholder
            raise ValueError(f"Error populating placeholder '{{{ph_full}}}': {str(e)}")
            
    return populated_receipt


if __name__ == '__main__':
    # Example usage for get_order_by_id (updated for new data):
    print("--- Testing get_order_by_id ---")
    try:
        order1 = get_order_by_id("ORD001")
        print(f"Found order: {order1}")

        order2 = get_order_by_id("ORD002")
        print(f"Found order: {order2}")

        # Example of a non-existent order
        order_non_existent = get_order_by_id("ORD999")
        print(f"Found order: {order_non_existent}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Testing check_template_placeholders (updated for new attributes) ---")
    # Example usage for check_template_placeholders:
    valid_template_1 = "Hello {customer.name}, your order {order.id} for ${order.total} is confirmed. Payment: {order.payment_method}."
    valid_template_2 = "Your items: {order.items}." # Changed to use order.items (list of strings)
    invalid_format_template = "Dear {customer_name}, your order total is {order.total}." # Missing dot
    invalid_object_template = "User: {user.name}, Order: {order.id}." # Invalid object 'user'
    invalid_attribute_template = "Order {order.id}, Status: {order.status}." # Invalid attribute 'status' for 'order'
    sensitive_template_ok_customer = "Customer: {customer.name}, Email: {customer.email}."
    sensitive_template_bad_customer = "Customer: {customer.name}, Secret: {customer.credit_card}." # Accessing sensitive customer data

    templates_to_test = {
        "Valid Template 1": valid_template_1,
        "Valid Template 2 (uses order.items)": valid_template_2,
        "Invalid Format Template": invalid_format_template,
        "Invalid Object Template": invalid_object_template,
        "Invalid Attribute Template (order.status)": invalid_attribute_template,
        "Valid Customer Info Template": sensitive_template_ok_customer,
        "Sensitive Data Template (customer.credit_card)": sensitive_template_bad_customer,
        "Invalid Attribute (order.date)": "Order date: {order.date}", # 'date' is no longer a valid attribute for order
        "Valid with unknown object (product)": "Details: {product.name}" # product not in VALID_OBJECT_ATTRIBUTES
    }

    for name, template_str in templates_to_test.items():
        print(f"\nTesting template: '{name}'")
        print(f"Template string: \"{template_str}\"")
        try:
            if check_template_placeholders(template_str):
                print("Result: Template is VALID.")
        except ValueError as e:
            print(f"Result: Template is INVALID. Reason: {e}")

    print("\n--- Testing generate_customer_receipt ---")
    receipt_template_valid = "Receipt for Order {order.id}\nCustomer: {customer.name} ({customer.email})\nItems: {order.items}\nTotal: ${order.total}\nPaid via: {order.payment_method}"
    receipt_template_sensitive = "Dear {customer.name}, your credit card {customer.credit_card} will be charged for order {order.id}."
    receipt_template_invalid_attr = "Order: {order.id}, Delivery ETA: {order.delivery_eta}" # delivery_eta is not valid

    test_receipt_cases = [
        ("ORD001", receipt_template_valid, "Valid Receipt Generation"),
        ("ORD002", receipt_template_valid, "Valid Receipt Generation (Order 2)"),
        ("ORD001", receipt_template_sensitive, "Sensitive Data in Receipt Template"),
        ("ORD001", receipt_template_invalid_attr, "Invalid Attribute in Receipt Template"),
        ("ORD999", receipt_template_valid, "Receipt for Non-existent Order")
    ]

    for order_id_test, template_test, case_name in test_receipt_cases:
        print(f"\nTesting receipt generation: '{case_name}' (Order ID: {order_id_test})")
        print(f"Template: \"{template_test}\"")
        try:
            receipt = generate_customer_receipt(order_id_test, template_test)
            print("Generated Receipt:")
            print(receipt)
        except ValueError as e:
            print(f"Error generating receipt: {e}")
