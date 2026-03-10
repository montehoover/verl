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
    Lets project owners add or remove team members from their own projects.
    Admins can manage any project without restrictions.

    Args:
        active_user_id: The ID of the user performing the task.
        proj_id: The ID of the project to be modified.
        operation: Can either be 'add' or 'remove'.
        target_member_id: The ID of the user to be added/removed.

    Returns:
        True if the modification is successful, otherwise False.
    """
    if proj_id not in PROJECTS:
        return False  # Project does not exist

    if active_user_id not in USERS:
        return False  # Active user does not exist
    
    # Target user must exist for both add and remove operations
    if target_member_id not in USERS:
        return False # Target user does not exist

    project = PROJECTS[proj_id]
    active_user = USERS[active_user_id]

    # Check permissions
    is_admin = active_user["role"] == "admin"
    is_owner = project["owner_id"] == active_user_id

    if not (is_admin or is_owner):
        return False  # User does not have permission

    if operation == "add":
        if target_member_id not in project["members"]:
            project["members"].append(target_member_id)
        return True  # Successfully added or already a member (idempotent)
    
    elif operation == "remove":
        # Owner cannot remove themselves
        if target_member_id == project["owner_id"]:
            return False
        
        if target_member_id in project["members"]:
            project["members"].remove(target_member_id)
            return True  # Successfully removed
        else:
            return False # Target member not in project, cannot remove
            
    else:
        return False  # Invalid operation
