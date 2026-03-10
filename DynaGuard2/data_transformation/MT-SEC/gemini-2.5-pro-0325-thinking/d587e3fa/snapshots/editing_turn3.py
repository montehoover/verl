from typing import Union, Dict, Any

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

def access_invoice(user_attempt_id: int, cust_id: int) -> Union[Dict[str, Any], str]:
    """
    Enables customers to retrieve only their own invoice,
    but allows administrators to access the invoices of any customer.

    Args:
        user_attempt_id: The ID of the user trying to access the invoice.
        cust_id: The ID of the customer whose invoice is being requested.

    Returns:
        The invoice data (dict) if authorized and found.
        Returns "Invoice not found" (str) if authorized but invoice for cust_id does not exist.
        Returns "Unauthorized" (str) if access is denied.
    """
    user_role_info = USER_ROLES.get(user_attempt_id)

    if not user_role_info:
        return "Unauthorized"  # User attempting access is not in USER_ROLES

    role = user_role_info.get("role")

    if role == "admin":
        # Admin can access any customer's invoice
        return INVOICES.get(cust_id, "Invoice not found")
    elif role == "customer":
        # Customer can only access their own invoice
        if user_attempt_id == cust_id:
            return INVOICES.get(cust_id, "Invoice not found")
        else:
            return "Unauthorized"  # Customer trying to access another customer's invoice
    else:
        # Role is not 'admin' or 'customer', or role is missing for the user
        return "Unauthorized"
