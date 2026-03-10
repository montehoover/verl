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

def is_admin_or_customer(user_id):
    if user_id in USER_ROLES:
        role = USER_ROLES[user_id]["role"]
        return role in ["admin", "customer"]
    return False

def can_access_invoice(user_id, customer_id):
    if user_id not in USER_ROLES:
        return False
    
    user_role = USER_ROLES[user_id]["role"]
    
    if user_role == "admin":
        return True
    elif user_role == "customer":
        return user_id == customer_id
    else:
        return False

def fetch_invoice(user_id, customer):
    if can_access_invoice(user_id, customer):
        if customer in INVOICES:
            return INVOICES[customer]
        else:
            return None
    else:
        return 'Unauthorized'
