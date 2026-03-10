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

def fetch_invoice(user_id: int, customer: int):
    """
    Fetches an invoice for a customer based on the requester's role.
    Admins can fetch any customer's invoice.
    Customers can only fetch their own invoice.

    Args:
        user_id (int): The ID of the user requesting the invoice.
        customer (int): The ID of the customer whose invoice is being fetched.

    Returns:
        dict or str: The invoice data if found and authorized,
                     'Invoice not found' if authorized but the invoice for the specified customer doesn't exist,
                     'Unauthorized' if the user_id is not authorized to perform the action
                     (e.g., requester not in USER_ROLES, customer trying to access another customer's invoice).
    """
    requester_info = USER_ROLES.get(user_id)

    if not requester_info:
        return 'Unauthorized'  # Requester's user_id not found in USER_ROLES

    requester_role = requester_info.get('role')
    is_authorized = False

    if requester_role == 'admin':
        is_authorized = True
    elif requester_role == 'customer':
        if user_id == customer:  # Customer can only fetch their own invoice
            is_authorized = True
    # For any other role or if role is not defined for the user, is_authorized remains False

    if is_authorized:
        invoice_data = INVOICES.get(customer)
        if invoice_data:
            return invoice_data
        else:
            # Authorized to request, but the specific customer's invoice doesn't exist
            return 'Invoice not found'
    else:
        # Requester is not authorized for this action
        return 'Unauthorized'
