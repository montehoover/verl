USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

def check_user_role(user_id: int) -> str:
    """
    Determines if a user is an admin, a project owner, or a regular user.

    Args:
        user_id: The ID of the user.

    Returns:
        A string indicating the user's role: 'admin', 'owner', or 'user'.
    """
    # Check if user is an admin
    user_data = USERS.get(user_id)
    if user_data and user_data.get("role") == "admin":
        return "admin"

    # Check if user is a project owner
    for project_details in PROJECTS.values():
        if project_details.get("owner_id") == user_id:
            return "owner"
            
    # If not admin and not owner, then considered a 'user'
    # This also covers users explicitly listed with role 'user' and not owners,
    # and users not found in USERS (they cannot be admin and if not owner, are 'user').
    return "user"

if __name__ == '__main__':
    # Example Usage:
    print(f"User 1 role: {check_user_role(1)}")  # Expected: admin
    print(f"User 2 role: {check_user_role(2)}")  # Expected: owner
    print(f"User 3 role: {check_user_role(3)}")  # Expected: owner
    print(f"User 4 role: {check_user_role(4)}")  # Expected: owner
    
    # Test case for a user who is 'user' in USERS and not an owner
    USERS[5] = {"user_id": 5, "role": "user"}
    print(f"User 5 role: {check_user_role(5)}")  # Expected: user

    # Test case for a user not in USERS and not an owner
    print(f"User 99 role: {check_user_role(99)}") # Expected: user
