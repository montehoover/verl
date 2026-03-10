from typing import Dict, Any

USERS: Dict[int, Dict[str, Any]] = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"},
}

PROJECTS: Dict[int, Dict[str, Any]] = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

# Toggle to allow project members (non-owners) to manage projects.
# By default, only admins and project owners can manage.
ALLOW_PROJECT_MEMBERS_TO_MANAGE: bool = False


def check_user_permission(user_id: int, project_id: int) -> bool:
    """
    Determine whether the given user has permission to manage the specified project.

    Rules:
    - Admins can manage any project.
    - Project owners can manage their own projects.
    - Optionally (if ALLOW_PROJECT_MEMBERS_TO_MANAGE is True), project members can manage.

    Args:
        user_id: The user's ID.
        project_id: The project's ID.

    Returns:
        True if the user can manage the project, otherwise False.
    """
    # Validate existence of user and project
    user = USERS.get(user_id)
    if user is None:
        return False

    project = PROJECTS.get(project_id)
    if project is None:
        return False

    # Admins can manage any project
    if user.get("role") == "admin":
        return True

    # Project owners can manage their own projects
    if project.get("owner_id") == user_id:
        return True

    # Optionally allow members to manage
    if ALLOW_PROJECT_MEMBERS_TO_MANAGE:
        members = project.get("members") or []
        if user_id in members:
            return True

    return False
