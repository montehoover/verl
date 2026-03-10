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


def _can_user_access_invoice(user_id: int, customer_id: int, user_roles: dict) -> bool:
    """
    Checks if a user has permission to access a specific customer's invoice.

    This is a pure function that determines access rights based on user roles
    and the relationship between the requesting user and the target customer.
    An administrator can access any customer's invoice.
    A customer can only access their own invoice.

    Args:
        user_id: The ID of the user requesting access.
        customer_id: The ID of the customer whose invoice is being targeted.
        user_roles: A dictionary mapping user IDs to their role information.
                    Expected format: {user_id: {"role": "role_name", ...}, ...}

    Returns:
        True if the user is authorized to access the invoice, False otherwise.
    """
    requesting_user = user_roles.get(user_id)

    if not requesting_user:
        return False  # User making the request does not exist

    user_role = requesting_user.get("role")

    if user_role == "admin":
        return True
    elif user_role == "customer":
        # A customer can only access their own invoice
        return user_id == customer_id
    
    return False # Unknown role or other cases not explicitly permitted


def fetch_invoice(user_id: int, customer: int):
    """
    Fetches an invoice for a customer based on user authorization.

    Args:
        user_id: The ID of the user requesting the invoice.
        customer: The ID of the customer whose invoice is being fetched.

    Returns:
        The relevant invoice data if the user is authorized, otherwise 'Unauthorized'.
    """
    if not _can_user_access_invoice(user_id, customer, USER_ROLES):
        return "Unauthorized"

    # User is authorized to attempt to fetch the invoice.
    invoice_data = INVOICES.get(customer)

    if invoice_data:
        return invoice_data
    else:
        # Invoice not found, but access was permitted.
        # Return a specific message based on the user's role.
        # _can_user_access_invoice ensures user_id is valid and has a role.
        requesting_user_role = USER_ROLES[user_id]["role"]

        if requesting_user_role == "admin":
            return "Invoice not found for customer {}".format(customer)
        elif requesting_user_role == "customer":
            # This implies user_id == customer, as per _can_user_access_invoice logic
            return "Invoice not found"
        
        # This part should ideally not be reached if _can_user_access_invoice 
        # and USER_ROLES are consistent and correctly define roles.
        # It's a fallback for unexpected states post-authorization.
        return "Error: Invoice not found after authorization, role unhandled."
