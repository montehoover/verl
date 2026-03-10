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

def handle_project_membership(acting_user_id: int, prjt_id: int, member_action: str, target_member_id: int) -> bool:
    """
    Manages project membership. Owners can add/remove members from their projects.
    Admins can manage any project.
    """
    # Validate inputs
    if acting_user_id not in USERS or target_member_id not in USERS:
        return False  # Invalid user ID
    if prjt_id not in PROJECTS:
        return False  # Invalid project ID
    if member_action not in ['add', 'remove']:
        return False  # Invalid action

    acting_user = USERS[acting_user_id]
    project = PROJECTS[prjt_id]

    # Check permissions
    is_admin = acting_user["role"] == "admin"
    is_owner = project["owner_id"] == acting_user_id

    if not (is_admin or is_owner):
        return False  # User does not have permission

    if member_action == 'add':
        if target_member_id not in project["members"]:
            project["members"].append(target_member_id)
            return True
        return False # Member already exists or other error

    elif member_action == 'remove':
        # Prevent owner from being removed
        if target_member_id == project["owner_id"]:
            return False

        if target_member_id in project["members"]:
            project["members"].remove(target_member_id)
            return True
        return False # Member not found or other error
    
    return False # Should not be reached if action is valid
