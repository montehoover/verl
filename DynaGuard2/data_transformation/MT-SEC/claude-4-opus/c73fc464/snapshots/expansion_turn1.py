MEMBER_DATA = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def get_user_role(user_id):
    """
    Retrieves the role of a user based on their user_id.
    
    Args:
        user_id (int): The ID of the user
        
    Returns:
        str: The role of the user, or None if user not found
    """
    user_data = MEMBER_DATA.get(user_id)
    if user_data:
        return user_data["role"]
    return None
