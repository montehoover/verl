import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def update_privileges(acting_user_id: int, affected_user_id: int) -> bool:
    """
    Checks if the acting user is an admin and logs the attempt to modify privileges.

    Args:
        acting_user_id: The ID of the user performing the action.
        affected_user_id: The ID of the user whose privileges are being affected.

    Returns:
        True if the acting user is an admin, False otherwise.
    """
    acting_user = USERS.get(acting_user_id)

    if acting_user and acting_user.get("role") == "admin":
        logging.info(
            f"Admin user {acting_user_id} is attempting to modify privileges for user {affected_user_id}."
        )
        return True
    else:
        if not acting_user:
            logging.warning(
                f"Attempt to update privileges by non-existent user {acting_user_id} for user {affected_user_id}."
            )
        else:
            logging.warning(
                f"User {acting_user_id} (role: {acting_user.get('role')}) "
                f"attempted to modify privileges for user {affected_user_id} without admin rights."
            )
        return False

if __name__ == '__main__':
    # Example Usage
    print(f"Admin attempt (user 1 on user 2): {update_privileges(1, 2)}")
    print(f"Non-admin attempt (user 2 on user 3): {update_privileges(2, 3)}")
    print(f"Moderator attempt (user 4 on user 2): {update_privileges(4, 2)}")
    print(f"Non-existent admin attempt (user 5 on user 2): {update_privileges(5, 2)}")
