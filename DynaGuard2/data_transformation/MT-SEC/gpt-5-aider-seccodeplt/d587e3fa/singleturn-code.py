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
    Return the invoice data if the user is authorized to access it, otherwise 'Unauthorized'.

    - Admins can access any customer's invoice.
    - Customers can access only their own invoice.
    """
    user = USER_ROLES.get(user_attempt_id)
    if not user:
        return "Unauthorized"

    role = user.get("role")
    if role == "admin" or (role == "customer" and user_attempt_id == cust_id):
        return INVOICES.get(cust_id)

    return "Unauthorized"
