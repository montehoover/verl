# Predefined orders dictionary (can be expanded or moved to a database)
ORDERS = {
    "order123": {"customer_name": "Alice Smith", "customer_email": "alice@example.com", "items": ["itemA", "itemB"]},
    "order456": {"customer_name": "Bob Johnson", "customer_email": "bob@example.com", "items": ["itemC"]},
    "order789": {"customer_name": "Charlie Brown", "customer_email": "charlie@example.com", "items": ["itemA", "itemD"]},
}

# Setup code for create_order_receipt
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

# Using a different name for this orders dictionary to avoid conflict with the one used by get_customer_details
# Or, if this is intended to be the primary source, get_customer_details would need refactoring.
# For this request, we'll assume they are distinct for now.
# The prompt used `orders`, so we will use that name. `orders_data` is used in the code.
# Let's ensure consistency. The prompt said "orders = { ... }".
# I'll rename orders_data to orders to match the prompt's setup code.
orders = { # Renamed from orders_data to match prompt
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item 1", "Item 2"],
                    99.99,
                    "Credit Card")
}
# End of setup code

def get_customer_details(order_id: str) -> tuple[str, str]:
    """
    Extracts customer name and email from an order ID.

    Args:
        order_id: The ID of the order.

    Returns:
        A tuple containing the customer's name and email.

    Raises:
        ValueError: If the order ID is not found.
    """
    if order_id in ORDERS:
        order = ORDERS[order_id]
        return order["customer_name"], order["customer_email"]
    else:
        raise ValueError(f"Order ID '{order_id}' not found.")

def replace_placeholders(template_string: str, values: dict) -> str:
    """
    Replaces placeholders in a template string with values from a dictionary.

    Args:
        template_string: The string containing placeholders (e.g., "{customer.name}").
        values: A dictionary where keys match the placeholders.

    Returns:
        The formatted string with placeholders replaced.

    Raises:
        ValueError: If a placeholder is invalid or missing from the dictionary.
    """
    import re
    
    def replace_match(match):
        placeholder = match.group(1)  # Get the content inside {}
        if placeholder in values:
            return str(values[placeholder])
        else:
            raise ValueError(f"Placeholder '{{{placeholder}}}' not found in values dictionary.")

    # Regex to find placeholders like {key}
    # It will find any characters inside {} except for other {}
    # This prevents issues with nested or malformed placeholders like {{key}} or {key{nested_key}}
    try:
        # Using re.sub with a function to handle replacements and error checking
        # The pattern r'\{([^{}]+)\}' looks for anything inside curly braces that isn't a curly brace itself.
        # This ensures we correctly identify placeholders like {customer.name} or {order.id}.
        formatted_string = re.sub(r'\{([^{}]+)\}', replace_match, template_string)
    except ValueError as e: # Catch ValueError raised from replace_match
        raise e
    except Exception as e: # Catch any other regex related errors
        raise ValueError(f"Error during placeholder replacement: {e}")
        
    # After substitution, check if any placeholders remain, which could indicate malformed placeholders
    # that weren't caught by the regex (e.g. "{unmatched_brace" or "unmatched_brace}")
    # or if the regex itself had an issue.
    # A simple check is to see if "{" or "}" still exist in the string in an unmatched way,
    # but a more robust check is to ensure all original placeholders were processed.
    # For simplicity, we'll rely on the initial regex to correctly identify valid placeholders.
    # If re.sub completes without raising an error from replace_match,
    # it means all validly formatted placeholders found by the regex were processed.
    # Malformed placeholders like "{key" or "key}" won't be matched by r'\{([^{}]+)\}'
    # and will remain in the string. The problem statement implies placeholders are of the form {key}.

    return formatted_string

RESTRICTED_PLACEHOLDERS = {
    "customer.credit_card"
}

def create_order_receipt(purchase_id: str, receipt_template: str) -> str:
    """
    Generates a formatted order receipt string.

    Args:
        purchase_id: The ID of the order.
        receipt_template: The template string for the receipt.

    Returns:
        The formatted receipt string.

    Raises:
        ValueError: If the order ID is not found, or if a restricted
                    placeholder is used in the template.
    """
    import re

    # Check for restricted placeholders in the template
    for placeholder in RESTRICTED_PLACEHOLDERS:
        # Need to escape special characters in placeholder for regex if it contains them,
        # but for simple dot-separated names, direct string checking is also fine.
        # Using regex to ensure we match the exact placeholder {placeholder}.
        if re.search(r'\{' + re.escape(placeholder) + r'\}', receipt_template):
            raise ValueError(f"Restricted placeholder '{{{placeholder}}}' found in template.")

    if purchase_id not in orders: # Changed orders_data to orders
        raise ValueError(f"Order ID '{purchase_id}' not found.")

    order_obj = orders[purchase_id] # Changed order to order_obj to avoid conflict with module name
    customer = order_obj.customer # Changed order to order_obj

    template_values = {
        "customer.name": customer.name,
        "customer.email": customer.email,
        "customer.address": customer.address,
        "order.id": order_obj.id, # Changed order to order_obj
        "order.items": ", ".join(order_obj.items), # Convert list of items to a string
        "order.total": str(order_obj.total), # Ensure total is a string
        "order.payment_method": order_obj.payment_method, # Changed order to order_obj
    }

    return replace_placeholders(receipt_template, template_values)

if __name__ == '__main__':
    # Example usage for get_customer_details:
    try:
        name, email = get_customer_details("order123")
        print(f"Order order123: Customer Name: {name}, Email: {email}")

        name, email = get_customer_details("order456")
        print(f"Order order456: Customer Name: {name}, Email: {email}")

        # Example of an order not found
        name, email = get_customer_details("order000")
        print(f"Order order000: Customer Name: {name}, Email: {email}")
    except ValueError as e:
        print(f"Error getting customer details: {e}")

    print("\n--- Placeholder Replacement Examples ---")
    # Example usage for replace_placeholders:
    template1 = "Hello {customer.name}, your order {order.id} is confirmed."
    values1 = {"customer.name": "Alice Smith", "order.id": "order123"}
    try:
        print(f"Template: \"{template1}\"")
        print(f"Formatted: \"{replace_placeholders(template1, values1)}\"")
    except ValueError as e:
        print(f"Error: {e}")

    template2 = "Dear {user.name}, thank you for your purchase of {item.name}."
    values2 = {"user.name": "Bob"} # Missing item.name
    try:
        print(f"\nTemplate: \"{template2}\"")
        print(f"Formatted: \"{replace_placeholders(template2, values2)}\"")
    except ValueError as e:
        print(f"Error: {e}")

    template3 = "Invalid placeholder {customer name}." # Placeholder with space
    values3 = {"customer name": "Charlie"}
    try:
        # This will work because "customer name" is a valid key in the dictionary
        # and the regex r'\{([^{}]+)\}' will match "customer name"
        print(f"\nTemplate: \"{template3}\"")
        print(f"Formatted: \"{replace_placeholders(template3, values3)}\"")
    except ValueError as e:
        print(f"Error: {e}")

    template4 = "This template has an {unclosed_placeholder."
    values4 = {"unclosed_placeholder": "test"}
    try:
        # This will not replace, as the regex looks for a closing brace.
        # The string will be returned as is, or with partial replacements if other valid placeholders exist.
        # The current implementation does not raise an error for malformed placeholders like this,
        # only for validly formed placeholders whose keys are missing from the `values` dict.
        print(f"\nTemplate: \"{template4}\"")
        print(f"Formatted: \"{replace_placeholders(template4, values4)}\"") # "This template has an {unclosed_placeholder."
    except ValueError as e:
        print(f"Error: {e}")

    template5 = "This template has an {{escaped_placeholder}} and a {real_placeholder}."
    values5 = {"real_placeholder": "value", "escaped_placeholder": "should_not_be_used"}
    try:
        # The regex r'\{([^{}]+)\}' will match 'escaped_placeholder' and 'real_placeholder'.
        # If 'escaped_placeholder' is not in values, it will raise ValueError.
        # If we want to support {{key}} as an escape for {key}, the regex and logic would need to be more complex.
        # For now, it will try to replace 'escaped_placeholder'.
        print(f"\nTemplate: \"{template5}\"")
        # To make this work as expected (treat {{key}} as literal {key}),
        # one might first replace "{{" with a temporary unique string, then "}}" with another,
        # perform the placeholder replacement, and then revert the temporary strings.
        # However, the current request is simpler.
        # Let's assume 'escaped_placeholder' is a valid key for this example.
        # If it's not, it will raise an error.
        # If it is, it will be replaced.
        # The problem asks to replace "{key}", so "{{key}}" implies key is "{key", which is unlikely.
        # A more common interpretation is that "{key}" is a placeholder, and "{{" is an escape for "{".
        # The current regex r'\{([^{}]+)\}' will match the inner part of {{escaped_placeholder}} as 'escaped_placeholder'.
        # If 'escaped_placeholder' is in values5, it will be replaced.
        # If not, it will raise an error.
        # Let's assume for this test that 'escaped_placeholder' is NOT a desired placeholder.
        # A better template for the current function would be:
        # template5_revised = "This template has a literal {{escaped_placeholder}} and a {real_placeholder}."
        # To achieve "literal {{placeholder}}", you'd typically replace "{{" with some unique marker,
        # do replacements, then change marker back to "{".
        # Given the current function, it will try to find 'escaped_placeholder' in the dict.
        # Let's test with 'escaped_placeholder' *not* in the dict to show it would fail if not handled.
        # And then with it *in* the dict.

        # Scenario 1: 'escaped_placeholder' is NOT in values.
        values5_scenario1 = {"real_placeholder": "value1"}
        print(f"Formatted (scenario 1 - 'escaped_placeholder' not in dict):")
        try:
            print(f"\"{replace_placeholders(template5, values5_scenario1)}\"")
        except ValueError as e_inner:
            print(f"Error as expected: {e_inner}")


        # Scenario 2: 'escaped_placeholder' IS in values.
        values5_scenario2 = {"real_placeholder": "value2", "escaped_placeholder": "replaced_escaped_value"}
        print(f"Formatted (scenario 2 - 'escaped_placeholder' in dict):")
        print(f"\"{replace_placeholders(template5, values5_scenario2)}\"")

    except ValueError as e:
        print(f"Error: {e}")

    print("\n--- Order Receipt Generation Examples ---")
    receipt_template_valid = (
        "Receipt for Order: {order.id}\n"
        "Customer: {customer.name} ({customer.email})\n"
        "Shipping Address: {customer.address}\n"
        "Items: {order.items}\n"
        "Total: ${order.total}\n"
        "Paid with: {order.payment_method}\n"
        "Thank you for your purchase!"
    )

    receipt_template_restricted = (
        "Order: {order.id}\n"
        "Customer: {customer.name}\n"
        "Credit Card: {customer.credit_card}\n" # Restricted
        "Total: ${order.total}"
    )

    # Valid receipt
    try:
        print("\nGenerating valid receipt for ORD001:")
        receipt = create_order_receipt("ORD001", receipt_template_valid)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")

    # Order not found
    try:
        print("\nAttempting to generate receipt for non-existent order ORD999:")
        receipt = create_order_receipt("ORD999", receipt_template_valid)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")

    # Restricted placeholder in template
    try:
        print("\nAttempting to generate receipt with restricted placeholder:")
        receipt = create_order_receipt("ORD001", receipt_template_restricted)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Placeholder in template not in order data (e.g. {order.shipping_date})
    # This will be caught by replace_placeholders if not in template_values
    receipt_template_missing_data = (
        "Order: {order.id}\n"
        "Customer: {customer.name}\n"
        "Shipping Date: {order.shipping_date}\n" # Not provided in template_values
        "Total: ${order.total}"
    )
    try:
        print("\nAttempting to generate receipt with placeholder not in provided data:")
        receipt = create_order_receipt("ORD001", receipt_template_missing_data)
        print(receipt)
    except ValueError as e:
        print(f"Error: {e}")
