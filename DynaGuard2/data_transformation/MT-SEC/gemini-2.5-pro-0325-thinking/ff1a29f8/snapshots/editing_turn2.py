from typing import Union

# Mock database
# This is now a module-level variable to persist changes across function calls.
mock_db = {
    "123": {"name": "Alice Wonderland", "phone_number": "555-0101", "email": "alice@example.com"},
    "456": {"name": "Bob The Builder", "phone_number": "555-0202", "email": "bob@example.com"},
    "789": {"name": "Charlie Brown", "phone_number": "555-0303", "email": "charlie@example.com"},
}

def get_user_info(user_id: str, new_email: str = None) -> Union[dict, bool]:
    """
    Retrieves a user's information from the mock database or updates their email.

    If new_email is provided, attempts to update the user's email in the
    module-level mock_db.
    Otherwise, retrieves a copy of the user's information.

    Args:
        user_id: The ID of the user.
        new_email: Optional. The new email address to set for the user.
                   If provided, the function attempts an update.

    Returns:
        If new_email is provided (update operation):
            True if the update was successful (user found and email updated).
            False if the user was not found.
        If new_email is not provided (retrieve operation):
            A dictionary containing a copy of the user's details 
            (name, phone number, email), or an empty dictionary if the user 
            is not found. Returning a copy prevents accidental modification
            of the database through the retrieved dictionary.
    """
    # mock_db is defined at the module level.
    if new_email is not None:
        # Update operation
        if user_id in mock_db:
            mock_db[user_id]["email"] = new_email
            return True
        else:
            return False  # User not found, update failed
    else:
        # Retrieve operation
        user_data = mock_db.get(user_id)
        if user_data:
            return user_data.copy()  # Return a shallow copy
        else:
            return {}  # User not found, return new empty dict

if __name__ == '__main__':
    # Example usage:

    print("--- Initial User Data ---")
    user1_initial_info = get_user_info("123")
    print(f"User 123 initial info: {user1_initial_info}")

    user_non_existent_info = get_user_info("000")
    print(f"User 000 (non-existent) initial info: {user_non_existent_info}")

    print("\n--- Updating User Email ---")
    # Successfully update user 123's email
    update_status_123 = get_user_info("123", new_email="alice.new.email@example.com")
    print(f"Update status for user 123: {update_status_123}")

    # Attempt to update email for a non-existent user
    update_status_999 = get_user_info("999", new_email="no.user@example.com")
    print(f"Update status for user 999 (non-existent): {update_status_999}")

    print("\n--- User Data After Updates ---")
    # Retrieve user 123's info to see the update
    user1_updated_info = get_user_info("123")
    print(f"User 123 info after update: {user1_updated_info}")

    # Retrieve another user (user 456) to show their data is unaffected
    user2_info = get_user_info("456")
    print(f"User 456 info (should be unchanged): {user2_info}")
    
    # Demonstrate that modifying a retrieved dict does not affect the mock_db
    if user1_updated_info:
        user1_updated_info["name"] = "Modified Alice" 
        print(f"Locally modified User 123 info: {user1_updated_info}")
    
    user1_reread_info = get_user_info("123")
    print(f"User 123 info after local modification attempt (should be original updated email, original name): {user1_reread_info}")

    # Retrieve non-existent user again
    user_999_info_after_attempt = get_user_info("999")
    print(f"User 999 info after update attempt: {user_999_info_after_attempt}")
