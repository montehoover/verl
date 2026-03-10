# Global data structures for user roles and invoices
user_roles = {}
user_invoices = {}

def can_view_own_invoice(user_id):
    """
    Check if a user can view their own invoice.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        bool: True if the user can view their invoice, False otherwise
    """
    # Check if user exists in the system
    if user_id not in user_roles:
        return False
    
    # Check if user has any invoices
    if user_id not in user_invoices:
        return False
    
    # Check if user has at least one invoice
    if len(user_invoices[user_id]) == 0:
        return False
    
    return True

def view_invoice(requestor_id, account_holder_id):
    """
    Allow viewing invoices for other users based on permissions.
    
    Args:
        requestor_id (int): The ID of the user requesting to view the invoice
        account_holder_id (int): The ID of the user whose invoice is being requested
        
    Returns:
        dict or str: The invoice data if authorized, otherwise 'Unauthorized'
    """
    # Check if requestor exists in the system
    if requestor_id not in user_roles:
        return 'Unauthorized'
    
    # Check if account holder exists and has invoices
    if account_holder_id not in user_invoices:
        return 'Unauthorized'
    
    # Allow users to view their own invoices
    if requestor_id == account_holder_id:
        if can_view_own_invoice(requestor_id):
            return user_invoices[account_holder_id]
        else:
            return 'Unauthorized'
    
    # Check if requestor has admin or billing role
    requestor_role = user_roles.get(requestor_id)
    if requestor_role in ['admin', 'billing']:
        return user_invoices[account_holder_id]
    
    return 'Unauthorized'
