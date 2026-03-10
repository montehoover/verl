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

def is_admin(user_id):
    """Check if a user is an administrator.
    
    Args:
        user_id (int): The user ID to check
        
    Returns:
        bool: True if the user is an admin, False otherwise
    """
    user = USER_ROLES.get(user_id)
    if user:
        return user.get("role") == "admin"
    return False

def can_access_invoice(requestor_id, account_holder_id):
    """Check if a user can access another user's invoice.
    
    Args:
        requestor_id (int): The ID of the user requesting access
        account_holder_id (int): The ID of the account holder whose invoice is being accessed
        
    Returns:
        bool: True if the requestor can access the invoice, False otherwise
    """
    # Users can access their own invoices
    if requestor_id == account_holder_id:
        return True
    
    # Admins can access any invoice
    if is_admin(requestor_id):
        return True
    
    return False

def retrieve_invoice(requestor_id, account_holder_id):
    """Retrieve invoice data if the requestor has permission.
    
    Args:
        requestor_id (int): The ID of the user requesting the invoice
        account_holder_id (int): The ID of the account holder whose invoice is being requested
        
    Returns:
        dict or str: The invoice data if authorized, otherwise 'Unauthorized'
    """
    if can_access_invoice(requestor_id, account_holder_id):
        invoice = INVOICES.get(account_holder_id)
        if invoice:
            return invoice
        return None
    return 'Unauthorized'
