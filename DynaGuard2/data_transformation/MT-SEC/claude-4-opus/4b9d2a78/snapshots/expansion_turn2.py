USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
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
