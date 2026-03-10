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
    Determines if a user is an admin or a regular user.

    Args:
        user_id: The ID of the user.

    Returns:
        A string indicating the user's role ("admin" or "user").
        Returns "unknown" if the user_id is not found.
    """
    user_info = USERS.get(user_id)
    if user_info:
        return user_info.get("role", "unknown")
    return "unknown"

def can_modify_project(executor_user_id: int, proj_id: int) -> bool:
    """
    Checks if a user can modify a specific project.
    Admins can modify any project.
    Regular users can only modify projects they own.

    Args:
        executor_user_id: The ID of the user attempting the action.
        proj_id: The ID of the project.

    Returns:
        True if the user can modify the project, False otherwise.
    """
    user_role = check_user_role(executor_user_id)
    if user_role == "admin":
        return True

    project_info = PROJECTS.get(proj_id)
    if project_info:
        return project_info.get("owner_id") == executor_user_id
    
    return False # Project not found or user is not owner

if __name__ == '__main__':
    # Example usage for check_user_role:
    print(f"User 1 role: {check_user_role(1)}")
    print(f"User 2 role: {check_user_role(2)}")
    print(f"User 5 role: {check_user_role(5)}")  # Test with a non-existent user

    # Example usage for can_modify_project:
    print(f"\nUser 1 (admin) modifying project 101: {can_modify_project(1, 101)}")
    print(f"User 2 (owner) modifying project 101: {can_modify_project(2, 101)}")
    print(f"User 3 (not owner) modifying project 101: {can_modify_project(3, 101)}")
    print(f"User 2 (owner) modifying project 102 (not owner): {can_modify_project(2, 102)}")
    print(f"User 4 (owner) modifying project 103: {can_modify_project(4, 103)}")
    print(f"User 2 modifying non-existent project 999: {can_modify_project(2, 999)}")
