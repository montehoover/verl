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

def authorize_user_action(acting_user_id: int, project_identifier: int, modification: str) -> bool:
    """
    Authorizes a user to perform an action on a project.

    Args:
        acting_user_id: The ID of the user performing the action.
        project_identifier: The ID of the project.
        modification: A string describing the action (e.g., "add_member", "remove_member").
                      This is not strictly used in the current logic but included for future enhancements.

    Returns:
        True if the user is authorized, False otherwise.
    """
    # For now, any user who can manage the project can perform any action.
    # The 'modification' parameter can be used later for more granular control.
    return check_user_permission(acting_user_id, project_identifier)

if __name__ == '__main__':
    # Test cases for check_user_permission
    print(f"Admin (1) managing project 101: {check_user_permission(1, 101)}")  # Expected: True
    print(f"Owner (2) managing project 101: {check_user_permission(2, 101)}")  # Expected: True
    print(f"Non-owner (3) managing project 101: {check_user_permission(3, 101)}") # Expected: False
    print(f"Owner (3) managing project 102: {check_user_permission(3, 102)}")  # Expected: True
    print(f"User (2) managing non-existent project 999: {check_user_permission(2, 999)}") # Expected: False
    print(f"Non-existent user 99 managing project 101: {check_user_permission(99, 101)}") # Expected: False
    print(f"Admin (1) managing non-existent project 999: {check_user_permission(1, 999)}") # Expected: False

    print("\n# Test cases for authorize_user_action")
    # Admin (1) tries to add a member to project 101
    print(f"Admin (1) action 'add_member' on project 101: {authorize_user_action(1, 101, 'add_member')}")  # Expected: True
    # Owner (2) tries to remove a member from project 101
    print(f"Owner (2) action 'remove_member' on project 101: {authorize_user_action(2, 101, 'remove_member')}")  # Expected: True
    # Non-owner (3) tries to add a member to project 101
    print(f"Non-owner (3) action 'add_member' on project 101: {authorize_user_action(3, 101, 'add_member')}") # Expected: False
    # Owner (3) tries to modify project 102
    print(f"Owner (3) action 'modify_settings' on project 102: {authorize_user_action(3, 102, 'modify_settings')}")  # Expected: True
    # User (2) tries an action on a non-existent project 999
    print(f"User (2) action 'add_member' on non-existent project 999: {authorize_user_action(2, 999, 'add_member')}") # Expected: False
    # Non-existent user 99 tries an action on project 101
    print(f"Non-existent user (99) action 'add_member' on project 101: {authorize_user_action(99, 101, 'add_member')}") # Expected: False
