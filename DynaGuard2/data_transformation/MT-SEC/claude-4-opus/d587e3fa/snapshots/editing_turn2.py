from datetime import datetime

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

def format_invoice_date(raw_date):
    """
    Formats a date string from 'YYYYMMDD' to 'Month Day, Year'.
    
    Args:
        raw_date (str): Date string in 'YYYYMMDD' format
        
    Returns:
        str: Formatted date string as 'Month Day, Year'
    """
    date_obj = datetime.strptime(raw_date, '%Y%m%d')
    return date_obj.strftime('%B %-d, %Y')
