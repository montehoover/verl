# Predefined orders dictionary (can be expanded or moved to a database)
ORDERS = {
    "order123": {"customer_name": "Alice Smith", "customer_email": "alice@example.com", "items": ["itemA", "itemB"]},
    "order456": {"customer_name": "Bob Johnson", "customer_email": "bob@example.com", "items": ["itemC"]},
    "order789": {"customer_name": "Charlie Brown", "customer_email": "charlie@example.com", "items": ["itemA", "itemD"]},
}

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
