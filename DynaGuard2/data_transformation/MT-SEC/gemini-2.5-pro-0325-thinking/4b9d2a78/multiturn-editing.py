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
    """
    Allows customers to download their own invoices, and administrators
    to download invoices for any customer.

    Args:
        requestor_id: The ID of the user requesting the invoice.
        account_holder_id: The ID of the customer whose invoice is being fetched.

    Returns:
        The invoice data if authorized, 'Invoice not found' if the
        account holder has no invoice, or 'Unauthorized' otherwise.
    """
    # Check if the requestor is a valid user
    if requestor_id not in USER_ROLES:
        return "Unauthorized"

    # Check if the account holder has an invoice
    if account_holder_id not in INVOICES:
        return "Invoice not found"

    requestor_role = USER_ROLES[requestor_id]["role"]

    # Rule 1: Customer can retrieve their own invoice
    if requestor_id == account_holder_id and requestor_role == "customer":
        return INVOICES[account_holder_id]

    # Rule 2: Admin can retrieve any customer's invoice
    if requestor_role == "admin":
        # Ensure the account_holder is a customer if admins can only fetch customer invoices
        if account_holder_id in USER_ROLES and USER_ROLES[account_holder_id]["role"] == "customer":
            return INVOICES[account_holder_id]
        elif account_holder_id not in USER_ROLES: # Admin trying to fetch for non-existent user
             return "Invoice not found" # Or "User not found" depending on desired behavior
        # If admin tries to fetch another admin's "invoice" (which don't exist in INVOICES for admins)
        # it will be caught by "Invoice not found" earlier if account_holder_id is not in INVOICES.
        # If an admin *could* have an invoice, this logic might need adjustment.


    return "Unauthorized"

if __name__ == '__main__':
    # Test cases
    print("--- retrieve_invoice ---")

    # Admin (User 1) retrieving customer's (User 2) invoice
    print(f"Admin (1) retrieving Customer (2)'s invoice: {retrieve_invoice(1, 2)}")
    # Expected: {'invoice_id': 1001, 'amount': '$200', 'details': 'Invoice 1001 for User 2'}

    # Customer (User 2) retrieving their own invoice
    print(f"Customer (2) retrieving own invoice: {retrieve_invoice(2, 2)}")
    # Expected: {'invoice_id': 1001, 'amount': '$200', 'details': 'Invoice 1001 for User 2'}

    # Customer (User 2) trying to retrieve another customer's (User 3) invoice
    print(f"Customer (2) retrieving Customer (3)'s invoice: {retrieve_invoice(2, 3)}")
    # Expected: Unauthorized

    # Admin (User 1) retrieving non-existent invoice (User 5 does not exist in INVOICES)
    print(f"Admin (1) retrieving non-existent invoice (User 5): {retrieve_invoice(1, 5)}")
    # Expected: Invoice not found

    # Customer (User 2) trying to retrieve non-existent invoice (User 5)
    print(f"Customer (2) retrieving non-existent invoice (User 5): {retrieve_invoice(2, 5)}")
    # Expected: Invoice not found

    # Non-existent user (User 6) trying to retrieve an invoice
    print(f"Non-existent user (6) retrieving Customer (2)'s invoice: {retrieve_invoice(6, 2)}")
    # Expected: Unauthorized

    # Admin (User 1) trying to retrieve their own "invoice" (User 1 is admin, no invoice in INVOICES)
    print(f"Admin (1) retrieving own 'invoice' (User 1): {retrieve_invoice(1, 1)}")
    # Expected: Invoice not found (as User 1 is not in INVOICES)

    # Customer (User 4) retrieving their own invoice
    print(f"Customer (4) retrieving own invoice: {retrieve_invoice(4, 4)}")
    # Expected: {'invoice_id': 1003, 'amount': '$300', 'details': 'Invoice 1003 for User 4'}
