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

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def format_order_details(order_id, template_string):
    """
    Retrieves order details and formats them using a template string.

    Args:
        order_id (str): The ID of the order to retrieve.
        template_string (str): A string with placeholders like {order_id}, 
                               {customer_name}, {total_amount}.

    Returns:
        str: The formatted string with order details.

    Raises:
        ValueError: If the order ID does not exist.
    """
    if order_id not in orders:
        raise ValueError(f"Order ID '{order_id}' not found.")
    
    order = orders[order_id]
    
    # Prepare data for formatting
    # Sensitive fields like email, address, credit_card are not included.
    details = SafeDict({
        "order_id": order.id,
        "customer_name": order.customer.name,
        "total_amount": f"{order.total:.2f}"
        # Add other non-sensitive fields here if needed
    })
    
    return template_string.format_map(details)

if __name__ == '__main__':
    # Example usage:
    template1 = "Order: {order_id}, Customer: {customer_name}, Total: ${total_amount}"
    formatted_details1 = format_order_details("ORD001", template1)
    print(formatted_details1)

    template2 = "Details for order {order_id} - Customer: {customer_name}. Amount due: ${total_amount}. Items: {items_list}."
    formatted_details2 = format_order_details("ORD001", template2)
    print(formatted_details2) # Demonstrates graceful handling of {items_list}
    
    try:
        format_order_details("ORD002", template1)
    except ValueError as e:
        print(e)
