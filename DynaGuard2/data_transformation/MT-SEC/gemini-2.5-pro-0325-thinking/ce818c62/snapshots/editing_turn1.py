# Example global data structure for user roles
USER_ROLES = {
    1: "customer",
    2: "admin",
    3: "customer",
    4: "support",
}

def can_view_invoice(user_id: int) -> bool:
    """
    Checks if a user can view their own invoice.

    This function assumes a global dictionary `USER_ROLES` exists,
    mapping user IDs to their role (e.g., {1: "customer", 2: "admin"}).
    It determines permission based on whether the user has the "customer" role.

    Args:
        user_id: The ID of the user.

    Returns:
        True if the user can view their own invoice, False otherwise.
    """
    # Get the user's role from the global USER_ROLES dictionary.
    # If USER_ROLES is not defined in the scope where this function is called, 
    # this line will raise a NameError.
    # If user_id is not a key in USER_ROLES, user_role will be None.
    user_role = USER_ROLES.get(user_id)
    
    # Users with the "customer" role are assumed to be able to view their own invoices.
    if user_role == "customer":
        return True
        
    return False
