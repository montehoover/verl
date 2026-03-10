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
    # Check if requestor exists in USER_ROLES
    if requestor_id not in USER_ROLES:
        return "Unauthorized"
    
    # Get requestor's role
    requestor_role = USER_ROLES[requestor_id]["role"]
    
    # Check authorization
    if requestor_role == "admin":
        # Admins can access any invoice
        if account_holder_id in INVOICES:
            return INVOICES[account_holder_id]
        else:
            return "Unauthorized"
    elif requestor_role == "customer":
        # Customers can only access their own invoice
        if requestor_id == account_holder_id and account_holder_id in INVOICES:
            return INVOICES[account_holder_id]
        else:
            return "Unauthorized"
    else:
        return "Unauthorized"
