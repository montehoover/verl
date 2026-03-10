MEMBER_DATA = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def get_user_role(user_id: int) -> str | None:
    """
    Retrieves the role of a user based on their user_id.

    Args:
        user_id: The ID of the user.

    Returns:
        The role of the user as a string, or None if the user is not found.
    """
    user_info = MEMBER_DATA.get(user_id)
    if user_info:
        return user_info.get("role")
    return None

if __name__ == '__main__':
    # Example usage:
    print(f"Role of user 1: {get_user_role(1)}")
    print(f"Role of user 3: {get_user_role(3)}")
    print(f"Role of user 5 (non-existent): {get_user_role(5)}")
