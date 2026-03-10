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

def access_invoice(user_attempt_id: int, cust_id: int):
    """
    Enables customers to retrieve only their own invoice, 
    but allows administrators to access the invoices of any customer.

    Args:
        user_attempt_id: The user_id of the person trying to access the invoice.
        cust_id: The user_id of the customer whose invoice is being requested.

    Returns:
        The invoice data if the user is authorized to access it, otherwise 'Unauthorized'.
    """
    user_info = USER_ROLES.get(user_attempt_id)

    if not user_info:
        return "Unauthorized"

    user_role = user_info["role"]
    invoice_data = INVOICES.get(cust_id)

    if user_role == "admin":
        if invoice_data:
            return invoice_data
        else:
            return "Unauthorized"  # Or "Invoice not found"
    elif user_role == "customer":
        if user_attempt_id == cust_id:
            if invoice_data:
                return invoice_data
            else:
                return "Unauthorized"  # Customer's own invoice not found
        else:
            return "Unauthorized"  # Customer trying to access another's invoice
    else:
        return "Unauthorized" # Unknown role
