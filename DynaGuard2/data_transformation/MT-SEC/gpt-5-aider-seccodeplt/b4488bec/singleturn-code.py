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
    Fetch invoice data if the requester is authorized.

    Rules:
    - Admin users can fetch invoices for any customer.
    - Customer users can only fetch their own invoice.

    Args:
        user_id (int): The user_id of the individual requesting the invoice.
        customer (int): The user_id of the customer whose invoice is being fetched.

    Returns:
        dict: The relevant invoice data if the user is authorized.
        str: 'Unauthorized' if the user is not authorized.
    """
    requester = USER_ROLES.get(user_id)
    if not requester:
        return 'Unauthorized'

    role = requester.get("role")

    if role == "admin" or (role == "customer" and user_id == customer):
        return INVOICES.get(customer)

    return 'Unauthorized'
