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

def is_admin_or_customer(user_id: int) -> bool:
    """
    Checks if the user is an admin or a customer.

    Args:
        user_id: The ID of the user.

    Returns:
        True if the user is an admin or a customer, False otherwise.
    """
    user_data = USER_ROLES.get(user_id)
    if user_data:
        return user_data["role"] in ("admin", "customer")
    return False

def can_access_invoice(user_id: int, customer_id: int) -> bool:
    """
    Checks if a user can access a specific customer's invoice.

    Args:
        user_id: The ID of the user attempting access.
        customer_id: The ID of the customer whose invoice is being accessed.

    Returns:
        True if the user can access the invoice, False otherwise.
    """
    user_data = USER_ROLES.get(user_id)
    if not user_data:
        return False  # User not found

    role = user_data["role"]

    if role == "admin":
        return True  # Admins can access any invoice

    if role == "customer":
        # Customers can only access their own invoices
        return user_id == customer_id
    
    return False # Should not happen if is_admin_or_customer logic is sound

def fetch_invoice(user_id: int, customer_id: int):
    """
    Retrieves an invoice if the user has the necessary permissions.

    Args:
        user_id: The ID of the user attempting to fetch the invoice.
        customer_id: The ID of the customer whose invoice is to be fetched.

    Returns:
        The invoice data if access is permitted, otherwise 'Unauthorized'.
        Returns None if the invoice is not found for an authorized user.
    """
    if not can_access_invoice(user_id, customer_id):
        return 'Unauthorized'
    
    return INVOICES.get(customer_id)
