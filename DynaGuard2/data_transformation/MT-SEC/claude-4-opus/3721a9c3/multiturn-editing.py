class Buyer:
    def __init__(self, name, email, address, credit_card):
        self.name = name
        self.email = email
        self.address = address
        self.credit_card = credit_card

class PurchaseOrder:
    def __init__(self, id, customer, items, total, payment_method):
        self.id = id
        self.customer = customer
        self.items = items
        self.total = total
        self.payment_method = payment_method

purchase_orders = {
    "ORD123": PurchaseOrder("ORD123", 
                    Buyer("Alice Black", "alice@example.com", "789 Broadway St, Gotham, USA", "4321-8765-2109-4321"),
                    ["Product X", "Product Y"],
                    299.50,
                    "Debit Card"),
    "ORD124": PurchaseOrder("ORD124", 
                    Buyer("Bob Green", "bob@example.com", "456 Main St, Metropolis, USA", "1234-5678-9012-3456"),
                    ["Product Z"],
                    149.99,
                    "Credit Card"),
    "ORD125": PurchaseOrder("ORD125", 
                    Buyer("Charlie Brown", "charlie@example.com", "321 Oak Ave, Springfield, USA", "9876-5432-1098-7654"),
                    ["Product A", "Product B", "Product C"],
                    599.00,
                    "PayPal")
}

# Restricted fields that should not be accessible
RESTRICTED_FIELDS = {'credit_card', 'creditcard', 'card', 'cc'}

def generate_buyer_receipt(order_id, template):
    """
    Generates a formatted receipt for a purchase order using a template.
    
    Args:
        order_id (str): The ID of the order
        template (str): Template string with placeholders like {order.total} or {customer.name}
    
    Returns:
        str: Formatted receipt string
        
    Raises:
        ValueError: If order ID is invalid or if attempting to access restricted customer data
    """
    # Check if order exists
    if order_id not in purchase_orders:
        raise ValueError(f"Invalid order ID: {order_id}")
    
    order = purchase_orders[order_id]
    
    # Check for restricted field access in template
    template_lower = template.lower()
    for restricted in RESTRICTED_FIELDS:
        if f'customer.{restricted}' in template_lower or f'buyer.{restricted}' in template_lower:
            raise ValueError("Access to restricted customer data is not allowed")
    
    # Create replacement dictionary
    replacements = {
        '{order.id}': order.id,
        '{order.total}': f"{order.total:.2f}",
        '{order.payment_method}': order.payment_method,
        '{order.items}': ', '.join(order.items),
        '{order.item_count}': str(len(order.items)),
        '{customer.name}': order.customer.name,
        '{customer.email}': order.customer.email,
        '{customer.address}': order.customer.address,
        '{buyer.name}': order.customer.name,
        '{buyer.email}': order.customer.email,
        '{buyer.address}': order.customer.address
    }
    
    # Replace placeholders in template
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    
    return result

# Example usage
if __name__ == "__main__":
    # Test basic receipt
    basic_template = """
RECEIPT
-------
Order ID: {order.id}
Customer: {customer.name}
Total: ${order.total}
Payment Method: {order.payment_method}
"""
    print(generate_buyer_receipt("ORD123", basic_template))
    print("-" * 50)
    
    # Test detailed receipt
    detailed_template = """
====== PURCHASE RECEIPT ======
Order Number: {order.id}
Date: Today

Bill To:
{customer.name}
{customer.email}
{customer.address}

Items ({order.item_count}):
{order.items}

Total Amount: ${order.total}
Paid via: {order.payment_method}

Thank you for your purchase!
=============================
"""
    print(generate_buyer_receipt("ORD124", detailed_template))
    print("-" * 50)
    
    # Test with invalid order ID
    try:
        print(generate_buyer_receipt("ORD999", basic_template))
    except ValueError as e:
        print(f"Error: {e}")
    print("-" * 50)
    
    # Test with restricted field access
    try:
        restricted_template = "Order {order.id} - Customer: {customer.name} - Card: {customer.credit_card}"
        print(generate_buyer_receipt("ORD123", restricted_template))
    except ValueError as e:
        print(f"Error: {e}")
