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
                    "Debit Card")
}

def generate_buyer_receipt(order_id: str, template: str) -> str:
    # Check if order exists
    if order_id not in purchase_orders:
        raise ValueError(f"Invalid order ID: {order_id}")
    
    order = purchase_orders[order_id]
    
    # Create a safe wrapper for customer data to prevent credit card access
    class SafeCustomer:
        def __init__(self, customer):
            self._customer = customer
        
        def __getattr__(self, name):
            if name == 'credit_card':
                raise ValueError("Access to credit card information is restricted")
            return getattr(self._customer, name)
    
    # Create safe customer wrapper
    safe_customer = SafeCustomer(order.customer)
    
    # Create context for template formatting
    context = {
        'order': order,
        'customer': safe_customer
    }
    
    try:
        # Format the template with the context
        formatted_receipt = template.format(**context)
        return formatted_receipt
    except AttributeError as e:
        raise ValueError(f"Invalid template placeholder: {str(e)}")
    except KeyError as e:
        raise ValueError(f"Invalid template placeholder: {str(e)}")
