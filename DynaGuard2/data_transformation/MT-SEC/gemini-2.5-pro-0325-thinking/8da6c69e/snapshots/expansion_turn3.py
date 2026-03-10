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

def has_permission_to_manage(user_id: int, project_id: int) -> bool:
    """
    Checks if a user has permission to manage a specific project.

    Args:
        user_id: The ID of the user.
        project_id: The ID of the project.

    Returns:
        True if the user has permission to manage the project, False otherwise.
    """
    user = USERS.get(user_id)
    project = PROJECTS.get(project_id)

    if not user or not project:
        return False  # User or project not found

    # Admins can manage any project
    if user.get("role") == "admin":
        return True

    # Project owners can manage their projects
    if project.get("owner_id") == user_id:
        return True

    return False

def can_perform_action(acting_user_id: int, prjt_id: int, member_action: str) -> bool:
    """
    Determines if a user can perform a membership action on a project.

    Args:
        acting_user_id: The ID of the user performing the action.
        prjt_id: The ID of the project.
        member_action: The action to be performed (e.g., "add", "remove").
                       Currently, this parameter is not used in the logic
                       as any management action requires management permission.

    Returns:
        True if the action is allowable, False otherwise.
    """
    # For now, any membership action requires management permission.
    # The member_action parameter can be used in the future for more granular control.
    return has_permission_to_manage(acting_user_id, prjt_id)

def handle_project_membership(acting_user_id: int, prjt_id: int, member_action: str, target_member_id: int) -> bool:
    """
    Handles adding or removing members from a project.

    Args:
        acting_user_id: The ID of the user performing the action.
        prjt_id: The ID of the project.
        member_action: The action to perform ("add" or "remove").
        target_member_id: The ID of the member to be added or removed.

    Returns:
        True if the operation was successful, False otherwise.
    """
    if not can_perform_action(acting_user_id, prjt_id, member_action):
        return False

    project = PROJECTS.get(prjt_id)
    if not project:
        return False  # Project not found

    acting_user = USERS.get(acting_user_id)
    if not acting_user:
        return False # Acting user not found

    target_user = USERS.get(target_member_id)
    if not target_user:
        return False # Target user for action does not exist

    if member_action == "add":
        if target_member_id in project["members"]:
            return False  # Target user is already a member
        project["members"].append(target_member_id)
        return True
    elif member_action == "remove":
        if target_member_id not in project["members"]:
            return False  # Target user is not a member

        # Prevent non-admins from removing the project owner from the members list.
        # This also prevents an owner from removing themselves from the members list.
        if target_member_id == project["owner_id"] and acting_user.get("role") != "admin":
            return False

        project["members"].remove(target_member_id)
        return True
    else:
        return False  # Invalid member_action
