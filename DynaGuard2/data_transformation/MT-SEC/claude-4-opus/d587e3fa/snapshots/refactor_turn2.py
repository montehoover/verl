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

def get_user_role(user_id):
    """Get the role of a user by their ID."""
    user = USER_ROLES.get(user_id)
    return user["role"] if user else None

def is_authorized_to_access(user_role, user_id, target_customer_id):
    """Check if a user is authorized to access a specific customer's invoice."""
    if user_role == "admin":
        return True
    if user_role == "customer" and user_id == target_customer_id:
        return True
    return False

def get_invoice(customer_id):
    """Retrieve invoice data for a specific customer."""
    return INVOICES.get(customer_id)

def access_invoice(user_attempt_id, cust_id):
    # Get the user's role
    user_role = get_user_role(user_attempt_id)
    
    # Check if user exists
    if user_role is None:
        return "Unauthorized"
    
    # Check authorization
    if not is_authorized_to_access(user_role, user_attempt_id, cust_id):
        return "Unauthorized"
    
    # Get the invoice
    invoice = get_invoice(cust_id)
    return invoice if invoice else "Unauthorized"
