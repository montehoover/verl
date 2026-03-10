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
    # Check if the requesting user exists in USER_ROLES
    if user_id not in USER_ROLES:
        return 'Unauthorized'
    
    # Get the role of the requesting user
    user_role = USER_ROLES[user_id]["role"]
    
    # Check authorization
    if user_role == "admin":
        # Admins can access any invoice
        if customer in INVOICES:
            return INVOICES[customer]
        else:
            return 'Unauthorized'  # Invoice doesn't exist
    elif user_role == "customer":
        # Customers can only access their own invoice
        if user_id == customer and customer in INVOICES:
            return INVOICES[customer]
        else:
            return 'Unauthorized'
    else:
        return 'Unauthorized'
