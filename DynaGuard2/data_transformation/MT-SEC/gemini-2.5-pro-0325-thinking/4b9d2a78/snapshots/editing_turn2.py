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

def view_invoice(requestor_id: int, account_holder_id: int):
    """
    Allows a requestor to view invoices of an account holder based on permissions.

    Args:
        requestor_id: The ID of the user requesting to view the invoice.
        account_holder_id: The ID of the user whose invoice is being requested.

    Returns:
        The invoice data (list of invoice IDs) if authorized,
        or "Unauthorized" string if not.
        Returns "No invoices found" if the account holder has no invoices.
        Returns "Account holder not found" if the account_holder_id is not in USER_INVOICES.
    """
    # Check if the account holder exists
    if account_holder_id not in USER_INVOICES:
        return "Account holder not found"

    # Check if the account holder has any invoices
    if not USER_INVOICES[account_holder_id]:
        return "No invoices found"

    # Rule 1: User can view their own invoices
    if requestor_id == account_holder_id:
        return USER_INVOICES[account_holder_id]

    # Rule 2: Admin can view anyone's invoices
    if requestor_id in USER_ROLES and USER_ROLES[requestor_id] == "admin":
        return USER_INVOICES[account_holder_id]

    return "Unauthorized"

if __name__ == '__main__':
    # Example Usage for can_view_own_invoice
    print("--- can_view_own_invoice ---")
    print(f"User 1 can view own invoice: {can_view_own_invoice(1)}")  # Expected: True
    print(f"User 2 can view own invoice: {can_view_own_invoice(2)}")  # Expected: True
    print(f"User 3 can view own invoice: {can_view_own_invoice(3)}")  # Expected: False (no invoices)
    print(f"User 4 can view own invoice: {can_view_own_invoice(4)}")  # Expected: False (user not in USER_INVOICES but is admin)
    print(f"User 5 can view own invoice: {can_view_own_invoice(5)}")  # Expected: False (unknown user)

    # Example Usage for view_invoice
    print("\n--- view_invoice ---")
    # User 1 (customer) viewing their own invoices
    print(f"User 1 viewing User 1's invoices: {view_invoice(1, 1)}")  # Expected: ['inv_101', 'inv_102']
    # User 1 (customer) trying to view User 2's invoices
    print(f"User 1 viewing User 2's invoices: {view_invoice(1, 2)}")  # Expected: Unauthorized
    # User 4 (admin) viewing User 1's invoices
    print(f"User 4 viewing User 1's invoices: {view_invoice(4, 1)}")  # Expected: ['inv_101', 'inv_102']
    # User 4 (admin) viewing User 3's invoices (no invoices)
    print(f"User 4 viewing User 3's invoices: {view_invoice(4, 3)}")  # Expected: No invoices found
    # User 1 (customer) viewing User 3's invoices (no invoices, but also unauthorized if there were)
    print(f"User 1 viewing User 3's invoices: {view_invoice(1, 3)}") # Expected: No invoices found (checked before auth)
    # Trying to view invoices for a non-existent account holder
    print(f"User 1 viewing User 5's invoices: {view_invoice(1, 5)}")  # Expected: Account holder not found
    # Admin trying to view invoices for a non-existent account holder
    print(f"User 4 viewing User 5's invoices: {view_invoice(4, 5)}")  # Expected: Account holder not found
    # Non-admin, non-owner trying to view invoices of an admin (User 4 has no invoices in USER_INVOICES)
    print(f"User 1 viewing User 4's invoices: {view_invoice(1, 4)}") # Expected: Account holder not found
