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
    Enables customers to retrieve only their own invoice, 
    but allows administrators to access the invoices of any customer.

    Args:
        user_attempt_id: The user_id of the person trying to access the invoice.
        cust_id: The user_id of the customer whose invoice is being requested.

    Returns:
        The invoice data if the user is authorized to access it, otherwise 'Unauthorized'.
    """
    
    # Check if the user attempting access is a known user
    if user_attempt_id not in USER_ROLES:
        return "Unauthorized"

    user_info = USER_ROLES[user_attempt_id]
    user_role = user_info["role"]

    if user_role == "admin":
        # Admin can access any invoice, provided it exists
        if cust_id in INVOICES:
            return INVOICES[cust_id]
        else:
            # Invoice for cust_id does not exist
            return "Unauthorized"
    elif user_role == "customer":
        # Customer can only access their own invoice
        if user_attempt_id == cust_id:
            if cust_id in INVOICES:
                return INVOICES[cust_id]
            else:
                # Customer's own invoice does not exist (should ideally not happen if data is consistent)
                return "Unauthorized"
        else:
            # Customer trying to access another's invoice
            return "Unauthorized"
    else:
        # Unknown role or other unauthorized cases
        return "Unauthorized"

if __name__ == '__main__':
    # Example Usage:
    # Admin access
    print(f"Admin (1) accessing User 2's invoice: {access_invoice(1, 2)}")
    print(f"Admin (1) accessing User 3's invoice: {access_invoice(1, 3)}")
    print(f"Admin (1) accessing User 5's invoice (non-existent): {access_invoice(1, 5)}")

    # Customer access
    print(f"Customer (2) accessing own invoice: {access_invoice(2, 2)}")
    print(f"Customer (2) accessing User 3's invoice: {access_invoice(2, 3)}")
    print(f"Customer (3) accessing own invoice (non-existent in INVOICES for this example, if INVOICES was incomplete): {access_invoice(3, 3)}") # Assuming invoice 3 exists
    
    # Non-existent user trying to access
    print(f"User 5 (non-existent) accessing User 2's invoice: {access_invoice(5, 2)}")

    # Customer whose invoice doesn't exist (e.g. if user 4 was in USER_ROLES but not INVOICES)
    # For this example, user 4 has an invoice. Let's test a valid user whose invoice is missing.
    # Temporarily remove user 4's invoice for a test case:
    # original_invoice_4 = INVOICES.pop(4, None) 
    # print(f"Customer (4) accessing own invoice (temporarily removed): {access_invoice(4, 4)}")
    # if original_invoice_4: INVOICES[4] = original_invoice_4 # Restore
    
    # Test case where customer exists but their invoice doesn't
    # Let's assume user 1 is admin, but what if they were a customer without an invoice?
    # This scenario is covered if cust_id (which is user_attempt_id for customer) is not in INVOICES.
    # e.g. if USER_ROLES had user 5 as customer, but INVOICES has no entry for 5.
    USER_ROLES[5] = {"user_id": 5, "role": "customer"} # Add a customer without an invoice
    print(f"Customer (5) accessing own invoice (no invoice data): {access_invoice(5, 5)}")
    del USER_ROLES[5] # Clean up test data
