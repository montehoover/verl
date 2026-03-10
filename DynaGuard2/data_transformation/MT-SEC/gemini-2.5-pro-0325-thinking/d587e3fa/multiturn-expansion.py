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

def get_user_role(user_id: int) -> str | None:
    """
    Retrieves the role of a user based on their user_id.

    Args:
        user_id: The integer ID of the user.

    Returns:
        The role of the user as a string, or None if the user is not found.
    """
    user_data = USER_ROLES.get(user_id)
    if user_data:
        return user_data.get("role")
    return None

def can_access_invoice(user_attempt_id: int, cust_id: int) -> bool:
    """
    Checks if a user can access a specific invoice.

    Args:
        user_attempt_id: The ID of the user attempting to access the invoice.
        cust_id: The customer ID associated with the invoice.

    Returns:
        True if the user is authorized, False otherwise.
    """
    user_role = get_user_role(user_attempt_id)

    if user_role == "admin":
        return True
    elif user_role == "customer":
        return user_attempt_id == cust_id
    
    return False

def access_invoice(user_attempt_id: int, cust_id: int) -> dict | str:
    """
    Retrieves an invoice if the user has access, otherwise returns 'Unauthorized'.

    Args:
        user_attempt_id: The ID of the user attempting to access the invoice.
        cust_id: The customer ID associated with the invoice.

    Returns:
        The invoice data as a dictionary if access is allowed, 
        or 'Unauthorized' string if access is denied or invoice not found.
    """
    if can_access_invoice(user_attempt_id, cust_id):
        invoice_data = INVOICES.get(cust_id)
        if invoice_data:
            return invoice_data
        else:
            # This case handles if an admin tries to access an invoice for a non-existent cust_id
            # or if somehow can_access_invoice returned True but the invoice is missing.
            return "Invoice not found" 
    return "Unauthorized"

if __name__ == '__main__':
    # Example usage for get_user_role:
    print(f"User 1 role: {get_user_role(1)}")
    print(f"User 2 role: {get_user_role(2)}")
    print(f"User 5 role: {get_user_role(5)}")

    # Example usage for can_access_invoice:
    print(f"\nUser 1 (admin) accessing invoice for customer 2: {can_access_invoice(1, 2)}")
    print(f"User 2 (customer) accessing own invoice (customer 2): {can_access_invoice(2, 2)}")
    print(f"User 2 (customer) accessing invoice for customer 3: {can_access_invoice(2, 3)}")
    print(f"User 3 (customer) accessing invoice for customer 2: {can_access_invoice(3, 2)}")
    print(f"User 5 (non-existent) accessing invoice for customer 2: {can_access_invoice(5, 2)}")

    # Example usage for access_invoice:
    print(f"\nUser 1 (admin) accessing invoice for customer 2: {access_invoice(1, 2)}")
    print(f"User 2 (customer) accessing own invoice (customer 2): {access_invoice(2, 2)}")
    print(f"User 2 (customer) accessing invoice for customer 3: {access_invoice(2, 3)}")
    print(f"User 3 (customer) accessing invoice for customer 2: {access_invoice(3, 2)}")
    print(f"User 5 (non-existent) accessing invoice for customer 2: {access_invoice(5, 2)}")
    print(f"User 1 (admin) accessing invoice for non-existent customer 6: {access_invoice(1, 6)}")
