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
    Allows customers to download their own invoices and administrators to download any customer's invoice.

    Args:
        current_user_id (int): The user_id of the actor requesting to download the invoice.
        customer_id (int): The user_id of the customer whose invoice is requested.

    Returns:
        dict | str: The invoice data if authorized, else 'Unauthorized'.
    """
    actor = USER_ROLES.get(current_user_id)
    if actor is None:
        return 'Unauthorized'

    if actor.get("role") == "admin" or current_user_id == customer_id:
        return INVOICES.get(customer_id)

    return 'Unauthorized'
