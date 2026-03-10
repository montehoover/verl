# Predefined orders dictionary
orders = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75, "items": ["Laptop", "Mouse"]},
    "102": {"customer_name": "Bob Johnson", "total_amount": 45.50, "items": ["Keyboard"]},
    "103": {"customer_name": "Carol Williams", "total_amount": 200.00, "items": ["Monitor", "Webcam"]},
}

def print_order_details(order_id: str, format_string: str = "Order ID: {order_id}\n  Customer Name: {customer_name}\n  Total Amount: ${total_amount:.2f}"):
    """
    Retrieves order details and formats them using a template string.

    Args:
        order_id: The ID of the order to retrieve.
        format_string: A string with placeholders like {order_id}, 
                       {customer_name}, and {total_amount}.

    Returns:
        A formatted string with order details, or a message if the order
        or details are not found.
    """
    order = orders.get(order_id)
    if order:
        details = {
            "order_id": order_id,
            "customer_name": order.get("customer_name", "N/A"),
            "total_amount": order.get("total_amount", 0.0) # Default to 0.0 for formatting
        }
        # Ensure total_amount is float for formatting, even if N/A was initially considered
        if details["total_amount"] == "N/A": # Should not happen with 0.0 default
            details["total_amount"] = 0.0

        try:
            # Special handling for total_amount if it's "N/A" conceptually,
            # though our default is 0.0. If it were "N/A", direct formatting would fail.
            # We will format total_amount separately if it's not "N/A"
            formatted_total_amount = f"{details['total_amount']:.2f}" if details['total_amount'] != "N/A" else "N/A"
            
            # Create a dictionary for formatting that includes the pre-formatted total_amount
            format_values = {
                "order_id": details["order_id"],
                "customer_name": details["customer_name"],
                "total_amount": details["total_amount"] # Use raw value for .format
            }
            
            # If a field in format_string is not in format_values, .format() will error.
            # We can use a custom Formatter or be careful with format_string.
            # For simplicity, we assume format_string uses known keys.
            # To handle missing keys more gracefully in format_string.format(), one might need a custom Formatter
            # or pre-process the format_string.
            # Here, we rely on the provided details dictionary.
            
            # A more robust way to handle missing keys in the format string itself:
            temp_details = details.copy()
            if isinstance(temp_details["total_amount"], (int, float)):
                temp_details["total_amount_formatted"] = f"${temp_details['total_amount']:.2f}"
            else:
                temp_details["total_amount_formatted"] = "$N/A"

            # Adjust format_string to use total_amount_formatted if it expects currency
            # This example assumes format_string might be like: "Cust: {customer_name}, Total: {total_amount_formatted}"
            # Or, if format_string is "{total_amount:.2f}", it needs a float.

            return format_string.format(
                order_id=details["order_id"],
                customer_name=details["customer_name"],
                total_amount=details["total_amount"] # This will be formatted by the :.2f in the string
            )
        except KeyError as e:
            return f"Formatting error: Missing key {e} in order details or format string."
        except ValueError: # Handles if total_amount is not a number for :.2f
            # This case is less likely with the current data structure and defaults
            return f"Formatting error: Invalid value for total_amount for order {order_id}."

    else:
        # For a not found order, we can still try to format if the format_string allows
        # Or return a specific message. Let's return a specific message.
        return f"Order ID: {order_id} not found."

if __name__ == "__main__":
    # Default format string
    print(print_order_details("101"))
    # Custom format string
    custom_format = "Order Number: {order_id} | Customer: {customer_name} | Amount: ${total_amount:.2f}"
    print(print_order_details("102", custom_format))
    
    # Order not found
    print(print_order_details("104", custom_format))
    
    # Incomplete order
    orders["105"] = {"customer_name": "David Brown"} # total_amount is missing
    # Default format string for incomplete order
    print(print_order_details("105"))
    # Custom format string for incomplete order
    print(print_order_details("105", custom_format))

    orders["106"] = {"total_amount": 75.20} # customer_name is missing
    print(print_order_details("106", "ID: {order_id}, Cust: {customer_name}, Amt: ${total_amount:.2f}"))
