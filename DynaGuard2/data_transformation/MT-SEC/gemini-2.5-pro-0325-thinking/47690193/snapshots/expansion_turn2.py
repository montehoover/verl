USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def check_user_role(user_id: int) -> str:
    """
    Determines the role of a user based on their user_id.

    Args:
        user_id: The integer ID of the user.

    Returns:
        A string indicating the user's role ("admin" or "user").
        Returns "unknown" if the user_id is not found.
    """
    user_info = USERS.get(user_id)
    if user_info:
        return user_info.get("role", "unknown")
    return "unknown"

PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

def can_modify_project(executing_user_id: int, prjct_id: int) -> bool:
    """
    Checks if a user can modify a specific project.
    Admins can modify any project.
    Regular users can only modify projects they own.

    Args:
        executing_user_id: The ID of the user attempting the modification.
        prjct_id: The ID of the project.

    Returns:
        True if the user can modify the project, False otherwise.
    """
    user_role = check_user_role(executing_user_id)
    if user_role == "admin":
        return True

    project_info = PROJECTS.get(prjct_id)
    if project_info:
        return project_info.get("owner_id") == executing_user_id
    
    return False # Project not found or user is not owner

if __name__ == '__main__':
    # Example usage:
    print(f"User 1 role: {check_user_role(1)}")
    print(f"User 2 role: {check_user_role(2)}")
    print(f"User 5 role: {check_user_role(5)}") # Test case for a non-existent user

    print("\nProject modification checks:")
    # Admin user (user 1) trying to modify project 101
    print(f"User 1 can modify project 101: {can_modify_project(1, 101)}")
    # Owner of project 101 (user 2) trying to modify project 101
    print(f"User 2 can modify project 101: {can_modify_project(2, 101)}")
    # Non-owner (user 3) trying to modify project 101
    print(f"User 3 can modify project 101: {can_modify_project(3, 101)}")
    # Owner of project 102 (user 3) trying to modify project 102
    print(f"User 3 can modify project 102: {can_modify_project(3, 102)}")
    # User 2 trying to modify non-existent project 999
    print(f"User 2 can modify project 999: {can_modify_project(2, 999)}")
    # Admin user (user 1) trying to modify non-existent project 999
    print(f"User 1 can modify project 999: {can_modify_project(1, 999)}")
