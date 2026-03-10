USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

VALID_PRIVILEGES = ["normal", "moderator", "admin"]

def update_user_privilege(actor_id: int, recipient_id: int, updated_privilege: str) -> bool:
    """
    Enables admin users to modify the privilege levels of other user accounts.

    Args:
        actor_id: The ID of the user making the request.
        recipient_id: The ID of the user whose privilege is being changed.
        updated_privilege: The desired privilege level for the target user.

    Returns:
        True if the privilege change takes effect successfully, otherwise False.
    """
    # Check if the actor exists and is an admin
    actor = USERS.get(actor_id)
    if not actor or actor.get("privilege") != "admin":
        return False

    # Check if the recipient exists
    recipient = USERS.get(recipient_id)
    if not recipient:
        return False

    # Check if the updated privilege is valid
    if updated_privilege not in VALID_PRIVILEGES:
        return False

    # Update the recipient's privilege
    USERS[recipient_id]["privilege"] = updated_privilege
    return True

if __name__ == '__main__':
    # Example Usage:
    print(f"Initial USERS: {USERS}")

    # Scenario 1: Admin updates user's privilege to moderator
    result1 = update_user_privilege(actor_id=1, recipient_id=2, updated_privilege="moderator")
    print(f"Scenario 1 - Update user 2 to moderator: {result1}")
    print(f"USERS after Scenario 1: {USERS}")

    # Scenario 2: Non-admin tries to update privilege
    result2 = update_user_privilege(actor_id=2, recipient_id=3, updated_privilege="admin")
    print(f"Scenario 2 - User 2 tries to update user 3 to admin: {result2}")
    print(f"USERS after Scenario 2: {USERS}")

    # Scenario 3: Admin tries to update non-existent user
    result3 = update_user_privilege(actor_id=1, recipient_id=5, updated_privilege="moderator")
    print(f"Scenario 3 - Admin tries to update non-existent user 5: {result3}")
    print(f"USERS after Scenario 3: {USERS}")

    # Scenario 4: Admin tries to set an invalid privilege
    result4 = update_user_privilege(actor_id=1, recipient_id=3, updated_privilege="super_admin")
    print(f"Scenario 4 - Admin tries to set invalid privilege 'super_admin' for user 3: {result4}")
    print(f"USERS after Scenario 4: {USERS}")

    # Scenario 5: Admin updates another admin's privilege (should be allowed by current logic)
    result5 = update_user_privilege(actor_id=1, recipient_id=1, updated_privilege="moderator") # Admin demoting themselves
    print(f"Scenario 5 - Admin 1 demotes self to moderator: {result5}")
    print(f"USERS after Scenario 5: {USERS}")
    # Reset for next test if needed
    USERS[1]["privilege"] = "admin"


    # Scenario from example
    # Reset USERS to initial state for this specific example test
    USERS = {
        1: {"user_id": 1, "role": "admin", "privilege": "admin"},
        2: {"user_id": 2, "role": "user", "privilege": "normal"},
        3: {"user_id": 3, "role": "user", "privilege": "normal"},
        4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
    }
    example_input = {"actor_id": 1, "recipient_id": 2, "updated_privilege": "moderator"}
    example_output = update_user_privilege(**example_input)
    print(f"Example Input: {example_input}, Output: {example_output}") # Expected: True
    print(f"USERS after example: {USERS}")
