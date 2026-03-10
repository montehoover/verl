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
