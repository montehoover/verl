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

def get_user_role(user_id: int) -> str:
    """
    Retrieves the role of a user based on their user_id.

    Args:
        user_id: The ID of the user.

    Returns:
        The role of the user as a string, or None if the user is not found.
    """
    user_data = USER_ROLES.get(user_id)
    if user_data:
        return user_data.get("role")
    return None

def can_download_invoice(current_user_id: int, customer_id: int) -> bool:
    """
    Checks if a user is authorized to download an invoice.

    Args:
        current_user_id: The ID of the user attempting the download.
        customer_id: The ID of the customer whose invoice is being downloaded.

    Returns:
        True if authorized, False otherwise.
    """
    current_user_role = get_user_role(current_user_id)

    if current_user_role == "admin":
        return True
    
    if current_user_role == "customer" and current_user_id == customer_id:
        return True
    
    return False

def download_invoice(current_user_id: int, customer_id: int):
    """
    Allows a user to download an invoice if authorized.

    Args:
        current_user_id: The ID of the user attempting the download.
        customer_id: The ID of the customer whose invoice is being downloaded.

    Returns:
        Invoice data if authorized, or 'Unauthorized' otherwise.
    """
    if can_download_invoice(current_user_id, customer_id):
        invoice_data = INVOICES.get(customer_id)
        if invoice_data:
            return invoice_data
        else:
            return "Invoice not found"  # Or handle as appropriate
    else:
        return "Unauthorized"
