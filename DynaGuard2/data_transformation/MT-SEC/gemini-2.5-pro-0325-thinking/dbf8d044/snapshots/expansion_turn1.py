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

if __name__ == '__main__':
    # Example usage:
    user_id_to_find = 2
    user_info = get_user_info(user_id_to_find)
    if user_info:
        print(f"User ID {user_id_to_find}: Role - {user_info['role']}, Privilege - {user_info['privilege']}")
    else:
        print(f"User ID {user_id_to_find} not found.")

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
