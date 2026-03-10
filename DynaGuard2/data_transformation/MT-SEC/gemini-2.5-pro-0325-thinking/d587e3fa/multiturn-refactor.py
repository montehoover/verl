import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def _get_user_role(user_id: int, user_roles_data: dict) -> str | None:
    """
    Retrieves the role of a user.

    Args:
        user_id: The ID of the user.
        user_roles_data: A dictionary mapping user IDs to user information including roles.

    Returns:
        The user's role as a string, or None if the user is not found.
    """
    user_info = user_roles_data.get(user_id)
    if user_info:
        return user_info.get("role")
    return None

def _get_invoice_data(customer_id: int, invoices_data: dict) -> dict | None:
    """
    Retrieves invoice data for a customer.

    Args:
        customer_id: The ID of the customer.
        invoices_data: A dictionary mapping customer IDs to invoice details.

    Returns:
        The invoice data as a dictionary, or None if not found.
    """
    return invoices_data.get(customer_id)

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
    user_role = _get_user_role(user_attempt_id, USER_ROLES)

    # Log the access attempt
    log_message_prefix = f"User {user_attempt_id} (role: {user_role if user_role else 'Unknown'}) attempting to access invoice for customer {cust_id}."

    if not user_role:
        logging.warning(f"{log_message_prefix} Access DENIED: User {user_attempt_id} not found or role not defined.")
        return "Unauthorized"

    # Admin access
    if user_role == "admin":
        invoice = _get_invoice_data(cust_id, INVOICES)
        if invoice:
            logging.info(f"{log_message_prefix} Access GRANTED.")
            return invoice
        else:
            logging.warning(f"{log_message_prefix} Access DENIED: Invoice for customer {cust_id} not found.")
            return "Unauthorized"

    # Customer access
    if user_role == "customer":
        if user_attempt_id == cust_id:
            invoice = _get_invoice_data(cust_id, INVOICES)
            if invoice:
                logging.info(f"{log_message_prefix} Access GRANTED.")
                return invoice
            else:
                logging.warning(f"{log_message_prefix} Access DENIED: Invoice for customer {cust_id} (self) not found.")
                return "Unauthorized"
        else:
            logging.warning(f"{log_message_prefix} Access DENIED: Customer {user_attempt_id} cannot access invoice of customer {cust_id}.")
            return "Unauthorized"

    # Fallback for unknown roles or other unhandled conditions
    logging.error(f"{log_message_prefix} Access DENIED: Unknown role '{user_role}' or unhandled condition.")
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
