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


def is_authorized_to_access_invoice(requestor_id: int, account_holder_id: int) -> bool:
    """Check if requestor is authorized to access the invoice."""
    if requestor_id not in USER_ROLES:
        return False
    
    requestor_role = USER_ROLES[requestor_id]["role"]
    
    if requestor_role == "admin":
        return True
    elif requestor_role == "customer":
        return requestor_id == account_holder_id
    else:
        return False


def get_invoice_data(account_holder_id: int):
    """Retrieve invoice data for a given account holder."""
    return INVOICES.get(account_holder_id)


def retrieve_invoice(requestor_id: int, account_holder_id: int):
    if not is_authorized_to_access_invoice(requestor_id, account_holder_id):
        return "Unauthorized"
    
    invoice_data = get_invoice_data(account_holder_id)
    
    if invoice_data is None:
        return "Unauthorized"
    
    return invoice_data
