from collections import defaultdict

# Sample predefined purchase orders
# In a real application, this data would likely come from a database or an API.
purchase_orders = {
    "101": {"customer_name": "Alice Smith", "total_amount": 150.75, "items": ["Laptop", "Mouse"]},
    "102": {"customer_name": "Bob Johnson", "total_amount": 25.00, "items": ["Keyboard"]},
    "103": {"customer_name": "Carol Williams", "total_amount": 300.50, "items": ["Monitor", "Webcam"]},
    "104": {"customer_name": "David Brown", "total_amount": 75.20, "items": ["Headphones"]},
}

def get_formatted_order_details(order_id, format_template):
    """
    Retrieves order details for a given order ID and formats them using a template string.

    Args:
        order_id (str or int): The ID of the order to retrieve.
        format_template (str): A string template with placeholders like {order_id},
                               {customer_name}, {total_amount}, {total_amount_formatted}, {items}.

    Returns:
        str: The formatted string with order details, or a message if the order is not found.
    """
    order_id_str = str(order_id) # Ensure order_id is consistently a string for lookups and formatting
    order = purchase_orders.get(order_id_str)

    if order:
        # Prepare data for formatting, using defaultdict for graceful missing placeholders in template
        format_data = defaultdict(lambda: "N/A")  # Default for keys not explicitly set

        format_data['order_id'] = order_id_str

        customer_name = order.get("customer_name")
        if customer_name is not None:
            format_data['customer_name'] = customer_name
        # If customer_name is None, template will use "N/A" from defaultdict if {customer_name} is used

        total_amount = order.get("total_amount")
        if total_amount is not None:
            format_data['total_amount'] = total_amount  # Raw amount
            format_data['total_amount_formatted'] = f"${total_amount:.2f}" # Formatted amount
        # If total_amount is None, template will use "N/A" for {total_amount} and {total_amount_formatted}

        items = order.get("items")
        if items is not None:
            format_data['items'] = ", ".join(items) # Comma-separated string for items
        # If items is None, template will use "N/A" for {items}

        return format_template.format_map(format_data)
    else:
        # For a non-existent order, return a clear message.
        # Alternatively, one could try to format this message with the template too,
        # if the template is designed to handle it (e.g. by only using {order_id}).
        # format_data_not_found = defaultdict(lambda: "N/A");
        # format_data_not_found['order_id'] = order_id_str;
        # return format_template.format_map(format_data_not_found)
        # This would result in "Order: 105, Customer: N/A, Amount: N/A." which might be undesirable.
        return f"Order ID: {order_id_str} not found."

if __name__ == '__main__':
    # Example usage:
    template_basic = "Order: {order_id}, Customer: {customer_name}, Amount: {total_amount_formatted}."
    template_detailed = "Order ID: {order_id}\n  Customer: {customer_name}\n  Total: {total_amount_formatted}\n  Items: {items}\n  Shipping Status: {shipping_status}." # {shipping_status} will be N/A

    print("--- Basic Format Examples ---")
    print(get_formatted_order_details("101", template_basic))
    
    # Example with integer order_id input
    output_102 = get_formatted_order_details(102, template_basic) # Function handles int order_id
    print(output_102)

    # Example of an order not found
    print(get_formatted_order_details("105", template_basic))

    print("\n--- Detailed Format Examples (with placeholder for non-existent data) ---")
    print(get_formatted_order_details("103", template_detailed)) # {shipping_status} will render as "N/A"

    # Example: Test an order where 'customer_name' is missing from the source data
    print("\n--- Testing Order with Temporarily Missing Customer Name from Source ---")
    original_customer_name_104 = purchase_orders["104"].pop("customer_name", None) # Simulate missing data
    print(get_formatted_order_details("104", template_basic)) # {customer_name} should become "N/A"
    if original_customer_name_104 is not None:
        purchase_orders["104"]["customer_name"] = original_customer_name_104 # Restore data

    # Example: Test an order where 'total_amount' is missing from the source data
    print("\n--- Testing Order with Temporarily Missing Total Amount from Source ---")
    original_total_amount_101 = purchase_orders["101"].pop("total_amount", None) # Simulate missing data
    print(get_formatted_order_details("101", template_detailed)) # {total_amount} and {total_amount_formatted} should be "N/A"
    if original_total_amount_101 is not None:
        purchase_orders["101"]["total_amount"] = original_total_amount_101 # Restore data
