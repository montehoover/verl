USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def get_user_info(user_id: int) -> dict | None:
    """
    Retrieves user details given a user ID.

    Args:
        user_id: The ID of the user.

    Returns:
        A dictionary with the user's role and privilege level,
        or None if the user is not found.
    """
    user_data = USERS.get(user_id)
    if user_data:
        return {
            "role": user_data["role"],
            "privilege": user_data["privilege"]
        }
    return None

def can_change_privilege(actor_id: int) -> bool:
    """
    Checks if a user has the authority to modify others' privilege levels.

    Args:
        actor_id: The ID of the user attempting the action.

    Returns:
        True if the user can change privileges, False otherwise.
    """
    actor_info = get_user_info(actor_id)
    if actor_info and actor_info["role"] == "admin":
        return True
    return False

if __name__ == '__main__':
    # Example usage:
    user_id_to_find = 2
    user_info = get_user_info(user_id_to_find)
    if user_info:
        print(f"User ID {user_id_to_find}: Role - {user_info['role']}, Privilege - {user_info['privilege']}")
    else:
        print(f"User ID {user_id_to_find} not found.")

    # Example usage for can_change_privilege:
    actor_id_admin = 1
    actor_id_user = 2
    actor_id_non_existent = 99

    print(f"User ID {actor_id_admin} can change privilege: {can_change_privilege(actor_id_admin)}")
    print(f"User ID {actor_id_user} can change privilege: {can_change_privilege(actor_id_user)}")
    print(f"User ID {actor_id_non_existent} can change privilege: {can_change_privilege(actor_id_non_existent)}")

    user_id_to_find = 5
    user_info = get_user_info(user_id_to_find)
    if user_info:
        print(f"User ID {user_id_to_find}: Role - {user_info['role']}, Privilege - {user_info['privilege']}")
    else:
        print(f"User ID {user_id_to_find} not found.")
    
    user_id_to_find = 1
    user_info = get_user_info(user_id_to_find)
    if user_info:
        print(f"User ID {user_id_to_find}: Role - {user_info['role']}, Privilege - {user_info['privilege']}")
    else:
        print(f"User ID {user_id_to_find} not found.")
