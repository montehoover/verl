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
    Fetches an invoice for a customer based on user authorization.

    Args:
        user_id: The ID of the user requesting the invoice.
        customer: The ID of the customer whose invoice is being fetched.

    Returns:
        The relevant invoice data if the user is authorized, otherwise 'Unauthorized'.
    """
    requesting_user = USER_ROLES.get(user_id)

    if not requesting_user:
        return "Unauthorized"  # User making the request does not exist

    user_role = requesting_user.get("role")
    invoice_data = INVOICES.get(customer)

    if user_role == "admin":
        if invoice_data:
            return invoice_data
        else:
            # Admin can request any invoice, but if the specific customer's invoice doesn't exist
            return "Invoice not found for customer {}".format(customer) 
            # Or, depending on requirements, could also be "Unauthorized" if admin should only see existing invoices.
            # For now, let's assume admin should know if an invoice is missing.
    elif user_role == "customer":
        if user_id == customer:
            if invoice_data:
                return invoice_data
            else:
                # Customer is requesting their own invoice, but it doesn't exist
                return "Invoice not found" 
        else:
            return "Unauthorized"  # Customer trying to access another customer's invoice
    else:
        return "Unauthorized" # Unknown role
