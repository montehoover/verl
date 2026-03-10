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

def retrieve_invoice(requestor_id: int, account_holder_id: int):
    """
    Retrieves an invoice if the requestor is authorized.

    Args:
        requestor_id: The user_id of the individual requesting the invoice.
        account_holder_id: The user_id of the customer whose invoice is being fetched.

    Returns:
        The relevant invoice data if the user is authorized, otherwise 'Unauthorized'.
        Returns None if authorized but no invoice exists for the account_holder_id.
    """
    requestor_user_data = USER_ROLES.get(requestor_id)

    if not requestor_user_data:
        return "Unauthorized"  # Requestor ID not found

    requestor_role = requestor_user_data.get("role")

    is_admin = requestor_role == "admin"
    # A customer can only request their own invoice
    is_self_request = requestor_role == "customer" and requestor_id == account_holder_id

    if is_admin or is_self_request:
        # Authorized to fetch the invoice
        return INVOICES.get(account_holder_id)
    else:
        # Not authorized
        return "Unauthorized"
