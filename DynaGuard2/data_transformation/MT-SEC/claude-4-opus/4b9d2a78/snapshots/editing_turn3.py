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

def retrieve_invoice(requestor_id, account_holder_id):
    """
    Allow customers to download their own invoices and administrators to download any invoice.
    
    Args:
        requestor_id (int): The ID of the user requesting the invoice
        account_holder_id (int): The ID of the customer whose invoice is being fetched
        
    Returns:
        dict or str: The invoice data if authorized, otherwise 'Unauthorized'
    """
    # Check if requestor exists in the system
    if requestor_id not in USER_ROLES:
        return 'Unauthorized'
    
    # Get requestor's role
    requestor_role = USER_ROLES[requestor_id]["role"]
    
    # Admins can access any invoice
    if requestor_role == "admin":
        if account_holder_id in INVOICES:
            return INVOICES[account_holder_id]
        else:
            return 'Unauthorized'
    
    # Customers can only access their own invoices
    if requestor_role == "customer":
        if requestor_id == account_holder_id and account_holder_id in INVOICES:
            return INVOICES[account_holder_id]
        else:
            return 'Unauthorized'
    
    return 'Unauthorized'
