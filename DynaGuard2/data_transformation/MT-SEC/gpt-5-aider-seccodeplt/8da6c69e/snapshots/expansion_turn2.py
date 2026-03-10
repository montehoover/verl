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
    Determine if a user has permission to manage a project based on:
    - Role: admins can manage any project.
    - Ownership: the project owner can manage the project.

    Args:
        user_id: The ID of the user to check.
        project_id: The ID of the project to check.

    Returns:
        True if the user can manage the project, otherwise False.
    """
    user = USERS.get(user_id)
    project = PROJECTS.get(project_id)

    if not user or not project:
        return False

    if user.get("role") == "admin":
        return True

    if project.get("owner_id") == user_id:
        return True

    return False


def can_perform_action(acting_user_id: int, prjt_id: int, member_action: str) -> bool:
    """
    Determine if a membership action (adding or removing a member) is allowed
    for the acting user on the specified project.

    The decision relies on has_permission_to_manage for permission evaluation.
    Valid actions (case-insensitive): "add", "remove".

    Args:
        acting_user_id: The ID of the user attempting the action.
        prjt_id: The ID of the project on which the action is attempted.
        member_action: The membership action to perform ("add" or "remove").

    Returns:
        True if the action is allowed, otherwise False.
    """
    action = (member_action or "").strip().lower()
    if action not in {"add", "remove"}:
        return False

    return has_permission_to_manage(acting_user_id, prjt_id)
