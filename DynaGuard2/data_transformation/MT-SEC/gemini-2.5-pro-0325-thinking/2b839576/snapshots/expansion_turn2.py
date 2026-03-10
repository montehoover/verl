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
        A string indicating the user's role ("admin", "user", or "unknown").
    """
    user_info = USERS.get(user_id)
    if user_info:
        return user_info.get("role", "unknown")
    return "unknown"

def can_manage_project(user_id: int, prj_id: int) -> bool:
    """
    Checks if a user can manage a specific project.
    Admins can manage any project.
    Project owners can manage their projects.

    Args:
        user_id: The ID of the user.
        prj_id: The ID of the project.

    Returns:
        True if the user can manage the project, False otherwise.
    """
    user_role = check_user_role(user_id)
    if user_role == "admin":
        return True

    project_info = PROJECTS.get(prj_id)
    if project_info:
        if project_info.get("owner_id") == user_id:
            return True
    
    return False

if __name__ == '__main__':
    # Example usage for check_user_role:
    print(f"User 1 role: {check_user_role(1)}")
    print(f"User 2 role: {check_user_role(2)}")
    print(f"User 5 role: {check_user_role(5)}")

    # Example usage for can_manage_project:
    print(f"\nUser 1 (admin) can manage project 101: {can_manage_project(1, 101)}")
    print(f"User 2 (owner) can manage project 101: {can_manage_project(2, 101)}")
    print(f"User 2 (not owner) can manage project 102: {can_manage_project(2, 102)}")
    print(f"User 3 (owner) can manage project 102: {can_manage_project(3, 102)}")
    print(f"User 4 (not admin, not owner) can manage project 101: {can_manage_project(4, 101)}")
    print(f"User 2 can manage non-existent project 999: {can_manage_project(2, 999)}")
