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

def download_invoice(current_user_id: int, customer_id: int):
    """
    Allows customers to download their own invoices and administrators
    to download any customer's invoice.

    Args:
        current_user_id: The ID of the user requesting the download.
        customer_id: The ID of the customer whose invoice is requested.

    Returns:
        The invoice data if authorized, or 'Unauthorized' otherwise.
    """
    current_user_info = USER_ROLES.get(current_user_id)
    
    if not current_user_info:
        return "Unauthorized"  # Current user not found

    current_user_role = current_user_info.get("role")

    # Admin can download any invoice
    if current_user_role == "admin":
        invoice_data = INVOICES.get(customer_id)
        return invoice_data if invoice_data else "Invoice not found"

    # Customer can only download their own invoice
    if current_user_role == "customer":
        if current_user_id == customer_id:
            invoice_data = INVOICES.get(customer_id)
            return invoice_data if invoice_data else "Invoice not found"
        else:
            return "Unauthorized"
            
    return "Unauthorized" # Default for any other role or unhandled case
