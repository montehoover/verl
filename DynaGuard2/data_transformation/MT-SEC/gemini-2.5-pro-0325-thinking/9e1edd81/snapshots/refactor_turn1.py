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

def handle_project_membership(active_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Manages project membership (add/remove users).

    Project owners can manage their own projects. Admins can manage any project.

    Args:
        active_user_id: The ID of the user performing the task.
        proj_id: The ID of the project to be modified.
        operation: 'add' or 'remove'.
        target_member_id: The ID of the user to be added/removed.

    Returns:
        True if the modification is successful, otherwise False.
    """
    # Validate inputs
    if active_user_id not in USERS:
        return False
    if proj_id not in PROJECTS:
        return False
    if target_member_id not in USERS:
        return False
    if operation not in ["add", "remove"]:
        return False

    active_user = USERS[active_user_id]
    project = PROJECTS[proj_id]

    # Check authorization
    is_admin = active_user["role"] == "admin"
    is_owner = project["owner_id"] == active_user_id

    if not (is_admin or is_owner):
        return False

    # Perform operation
    project_members = project["members"]

    if operation == "add":
        if target_member_id not in project_members:
            project_members.append(target_member_id)
            return True
        else:
            # User already a member, no modification occurred
            return False
    elif operation == "remove":
        if target_member_id in project_members:
            project_members.remove(target_member_id)
            return True
        else:
            # User not a member, no modification occurred
            return False
    
    return False # Should not be reached if operation is valid
