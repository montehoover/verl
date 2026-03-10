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
    # Check if current user exists
    if current_user_id not in USER_ROLES:
        return 'Unauthorized'
    
    # Get the current user's role
    current_user_role = USER_ROLES[current_user_id]["role"]
    
    # Check authorization
    if current_user_role == "admin":
        # Admins can download any customer's invoice
        if customer_id in INVOICES:
            return INVOICES[customer_id]
        else:
            return 'Unauthorized'
    elif current_user_role == "customer":
        # Customers can only download their own invoice
        if current_user_id == customer_id and customer_id in INVOICES:
            return INVOICES[customer_id]
        else:
            return 'Unauthorized'
    else:
        return 'Unauthorized'
