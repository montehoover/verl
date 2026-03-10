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


def has_permission(user_id: int, proj_id: int) -> bool:
    """
    Determine if a user has permission to manage a project.

    Rules:
    - Admins can manage any project.
    - The project owner can manage their project.
    - Non-admin, non-owner users cannot manage the project.

    Args:
        user_id: The ID of the user.
        proj_id: The ID of the project.

    Returns:
        True if the user can manage the project, otherwise False.
    """
    user = USERS.get(user_id)
    project = PROJECTS.get(proj_id)

    if not user or not project:
        return False

    if user.get("role") == "admin":
        return True

    if project.get("owner_id") == user_id:
        return True

    return False


def can_perform_action(acting_user_id: int, prj_id: int, action_type: str) -> bool:
    """
    Determine if the acting user can perform a specific action on a project.

    Supported actions:
    - 'add_member'
    - 'remove_member'

    Logic:
    - Only users who can manage the project (per has_permission) are allowed
      to perform member management actions.

    Args:
        acting_user_id: The ID of the user attempting the action.
        prj_id: The ID of the target project.
        action_type: The action being attempted.

    Returns:
        True if the action is allowed, otherwise False.
    """
    action = str(action_type).strip().lower()
    if action not in {"add_member", "remove_member"}:
        return False

    return has_permission(acting_user_id, prj_id)
