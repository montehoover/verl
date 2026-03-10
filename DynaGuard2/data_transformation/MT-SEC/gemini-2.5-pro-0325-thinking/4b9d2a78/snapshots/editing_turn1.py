# Global data structure for user invoices
# Example: {user_id: [invoice_id1, invoice_id2, ...]}
USER_INVOICES = {
    1: ["inv_101", "inv_102"],
    2: ["inv_201"],
    3: [],  # User 3 has no invoices
}

# Global data structure for user roles (not strictly needed for this function, but mentioned)
# Example: {user_id: "role"}
USER_ROLES = {
    1: "customer",
    2: "customer",
    3: "customer",
    4: "admin",
}

def can_view_own_invoice(user_id: int) -> bool:
    """
    Checks if a user can view their own invoice.

    Args:
        user_id: The ID of the user.

    Returns:
        True if the user can view their own invoice, False otherwise.
    """
    # A user can view their own invoice if they have invoices associated with them.
    if user_id in USER_INVOICES and USER_INVOICES[user_id]:
        return True
    return False

if __name__ == '__main__':
    # Example Usage
    print(f"User 1 can view own invoice: {can_view_own_invoice(1)}")  # Expected: True
    print(f"User 2 can view own invoice: {can_view_own_invoice(2)}")  # Expected: True
    print(f"User 3 can view own invoice: {can_view_own_invoice(3)}")  # Expected: False (no invoices)
    print(f"User 4 can view own invoice: {can_view_own_invoice(4)}")  # Expected: False (user not in USER_INVOICES)
    print(f"User 5 can view own invoice: {can_view_own_invoice(5)}")  # Expected: False (unknown user)
