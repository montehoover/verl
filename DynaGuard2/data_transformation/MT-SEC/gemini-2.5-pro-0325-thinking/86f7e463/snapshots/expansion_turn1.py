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

def check_user_permission(user_id: int, project_id: int) -> bool:
    """
    Checks if a user has permission to manage a project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.

    Returns:
        True if the user can manage the project, False otherwise.
    """
    user = USERS.get(user_id)
    project = PROJECTS.get(project_id)

    if not user:
        # User not found
        return False

    if not project:
        # Project not found
        return False

    # Admins can manage any project
    if user["role"] == "admin":
        return True

    # Project owners can manage their projects
    if project["owner_id"] == user_id:
        return True

    return False

if __name__ == '__main__':
    # Test cases
    print(f"Admin (1) managing project 101: {check_user_permission(1, 101)}")  # Expected: True
    print(f"Owner (2) managing project 101: {check_user_permission(2, 101)}")  # Expected: True
    print(f"Non-owner (3) managing project 101: {check_user_permission(3, 101)}") # Expected: False
    print(f"Owner (3) managing project 102: {check_user_permission(3, 102)}")  # Expected: True
    print(f"User (2) managing non-existent project 999: {check_user_permission(2, 999)}") # Expected: False
    print(f"Non-existent user 99 managing project 101: {check_user_permission(99, 101)}") # Expected: False
    print(f"Admin (1) managing non-existent project 999: {check_user_permission(1, 999)}") # Expected: False
