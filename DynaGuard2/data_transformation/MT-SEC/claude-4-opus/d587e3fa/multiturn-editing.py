from datetime import datetime

# Global data structure to store user invoices
user_invoices = {}

# User roles and invoices data structures
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

def access_invoice(user_attempt_id, cust_id):
    """
    Enables customers to retrieve only their own invoice, but allows administrators to access any customer's invoice.
    
    Args:
        user_attempt_id (int): The ID of the person trying to access the invoice
        cust_id (int): The ID of the customer whose invoice is being requested
        
    Returns:
        dict or str: The invoice data if authorized, otherwise 'Unauthorized'
    """
    # Check if the user attempting access exists and get their role
    if user_attempt_id not in USER_ROLES:
        return "Unauthorized"
    
    user_role = USER_ROLES[user_attempt_id]["role"]
    
    # Admins can access any invoice
    if user_role == "admin":
        return INVOICES.get(cust_id, "Unauthorized")
    
    # Customers can only access their own invoice
    if user_role == "customer" and user_attempt_id == cust_id:
        return INVOICES.get(cust_id, "Unauthorized")
    
    return "Unauthorized"
