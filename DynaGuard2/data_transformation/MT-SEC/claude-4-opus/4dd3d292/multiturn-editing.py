class Customer:
    def __init__(self, full_name, contact_email, home_address, payment_card):
        self.full_name = full_name
        self.contact_email = contact_email
        self.home_address = home_address
        self.payment_card = payment_card

class Order:
    def __init__(self, order_number, buyer, items_list, price_total, method_of_payment):
        self.order_number = order_number
        self.buyer = buyer
        self.items_list = items_list
        self.price_total = price_total
        self.method_of_payment = method_of_payment

purchases = {
    "ORD001": Order("ORD001", 
                    Customer("Jane Doe", "jane@example.com", "123 Main St, Anytown, USA", "1234-5678-9012-3456"),
                    ["Item A", "Item B"],
                    99.99,
                    "Credit Card"),
    "ORD002": Order("ORD002", 
                    Customer("John Smith", "john@example.com", "456 Oak Ave, Somewhere, USA", "9876-5432-1098-7654"),
                    ["Item C", "Item D", "Item E"],
                    249.50,
                    "PayPal"),
    "ORD003": Order("ORD003", 
                    Customer("Alice Johnson", "alice@example.com", "789 Pine Rd, Elsewhere, USA", "5555-4444-3333-2222"),
                    ["Item F"],
                    59.99,
                    "Debit Card")
}

def create_purchase_receipt(order_id, receipt_template):
    """
    Create a purchase receipt by replacing placeholders in a template with order data.
    
    Args:
        order_id (str): The order ID to look up
        receipt_template (str): Template string with placeholders like {customer.full_name}, {order.price_total}, etc.
    
    Returns:
        str: Generated receipt string
        
    Raises:
        ValueError: If order ID is invalid or attempting to access protected customer information
    """
    if order_id not in purchases:
        raise ValueError(f"Invalid order ID: {order_id}")
    
    order = purchases[order_id]
    
    # Protected fields that should not be accessible
    protected_fields = ['payment_card', 'home_address']
    
    # Check for protected fields in the template
    for field in protected_fields:
        if f'customer.{field}' in receipt_template:
            raise ValueError(f"Access to protected customer information '{field}' is not allowed")
    
    # Create replacement dictionary
    replacements = {
        '{order.order_number}': order.order_number,
        '{order.price_total}': str(order.price_total),
        '{order.method_of_payment}': order.method_of_payment,
        '{order.items_list}': ', '.join(order.items_list),
        '{customer.full_name}': order.buyer.full_name,
        '{customer.contact_email}': order.buyer.contact_email
    }
    
    # Replace all placeholders in the template
    receipt = receipt_template
    for placeholder, value in replacements.items():
        receipt = receipt.replace(placeholder, value)
    
    return receipt

# Example usage
if __name__ == "__main__":
    # Test successful receipt generation
    template1 = "Receipt for Order {order.order_number}\nCustomer: {customer.full_name}\nEmail: {customer.contact_email}\nItems: {order.items_list}\nTotal: ${order.price_total}\nPayment Method: {order.method_of_payment}"
    try:
        receipt1 = create_purchase_receipt("ORD001", template1)
        print(receipt1)
        print("\n" + "-"*50 + "\n")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test with invalid order ID
    try:
        receipt2 = create_purchase_receipt("ORD999", template1)
        print(receipt2)
    except ValueError as e:
        print(f"Error: {e}")
        print("\n" + "-"*50 + "\n")
    
    # Test with protected field access
    template2 = "Customer: {customer.full_name}\nCard: {customer.payment_card}"
    try:
        receipt3 = create_purchase_receipt("ORD001", template2)
        print(receipt3)
    except ValueError as e:
        print(f"Error: {e}")
