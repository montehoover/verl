# Global data structures for user roles and invoices
USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
}

INVOICES = {
    2: {"invoice_id": 1001, "amount": "$200", "details": "Invoice 1001 for User 2"},
    3: {"invoice_id": 1002, "amount": "$150", "details": "Invoice 1002 for User 3"},
    4: {"invoice_id": 1003, "amount": "$300", "details": "Invoice 1003 for User 4"}
}

def can_view_invoice(user_id):
    """Check if a user can view their own invoice."""
    # Check if user exists in the system
    if user_id not in USER_ROLES:
        return False
    
    # Check if user has any invoices
    if user_id not in INVOICES:
        return False
    
    # User can view their own invoice if they exist and have invoices
    return True

def download_invoice(current_user_id, customer_id):
    """Allow customers to download their own invoices and administrators to download any customer's invoice."""
    # Check if current user exists
    if current_user_id not in USER_ROLES:
        return "Unauthorized"
    
    # Check if the requested customer has an invoice
    if customer_id not in INVOICES:
        return "Unauthorized"
    
    # Get current user's role
    current_user_role = USER_ROLES[current_user_id]["role"]
    
    # Admins can download any invoice
    if current_user_role == "admin":
        return INVOICES[customer_id]
    
    # Customers can only download their own invoice
    if current_user_role == "customer" and current_user_id == customer_id:
        return INVOICES[customer_id]
    
    return "Unauthorized"
