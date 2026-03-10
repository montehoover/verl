# Global data structures for user roles and invoices
user_roles = {}
user_invoices = {}

def can_view_invoice(user_id):
    """Check if a user can view their own invoice."""
    # Check if user exists in the system
    if user_id not in user_roles:
        return False
    
    # Check if user has any invoices
    if user_id not in user_invoices:
        return False
    
    # User can view their own invoice if they exist and have invoices
    return True

def format_invoice_amount(amount):
    """Convert invoice amount string to float."""
    # Remove dollar sign and any whitespace
    cleaned_amount = amount.strip().replace('$', '')
    
    # Convert to float
    return float(cleaned_amount)
