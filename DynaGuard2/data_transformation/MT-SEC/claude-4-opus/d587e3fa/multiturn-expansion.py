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

def get_user_role(user_id):
    if user_id in USER_ROLES:
        return USER_ROLES[user_id]["role"]
    return None

def can_access_invoice(user_attempt_id, cust_id):
    role = get_user_role(user_attempt_id)
    if role == "admin":
        return True
    elif role == "customer" and user_attempt_id == cust_id:
        return True
    return False

def access_invoice(user_attempt_id, cust_id):
    if can_access_invoice(user_attempt_id, cust_id):
        if cust_id in INVOICES:
            return INVOICES[cust_id]
        else:
            return None
    return "Unauthorized"
