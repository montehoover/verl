# Global data structure to store user invoices
user_invoices = {}

def view_own_invoice(user_id):
    """
    Allows a user to view their own invoice.
    
    Args:
        user_id (int): The ID of the user requesting their invoice
        
    Returns:
        dict: The invoice details for the user, or None if no invoice exists
    """
    return user_invoices.get(user_id)
