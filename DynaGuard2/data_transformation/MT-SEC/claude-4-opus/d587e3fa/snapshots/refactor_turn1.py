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

def access_invoice(user_attempt_id, cust_id):
    # Check if the user exists in USER_ROLES
    if user_attempt_id not in USER_ROLES:
        return "Unauthorized"
    
    # Get the user's role
    user_role = USER_ROLES[user_attempt_id]["role"]
    
    # If user is admin, they can access any invoice
    if user_role == "admin":
        return INVOICES.get(cust_id, "Unauthorized")
    
    # If user is customer, they can only access their own invoice
    if user_role == "customer" and user_attempt_id == cust_id:
        return INVOICES.get(cust_id, "Unauthorized")
    
    # Otherwise, unauthorized
    return "Unauthorized"
