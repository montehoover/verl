from typing import Dict, List, Literal

USERS: Dict[int, Dict[str, object]] = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"},
}

PROJECTS: Dict[int, Dict[str, object]] = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}

RoleStr = Literal["admin", "owner", "user"]


def check_user_role(user_id: int) -> RoleStr:
    """
    Determine a user's role across the system.

    Priority:
    - "admin" if the user has role "admin" in USERS.
    - "owner" if the user owns at least one project in PROJECTS.
    - "user" otherwise.

    If the user_id does not exist in USERS, returns "user".

    Args:
        user_id: The ID of the user to check.

    Returns:
        One of "admin", "owner", or "user".
    """
    user = USERS.get(user_id)
    if not user:
        return "user"

    if str(user.get("role")) == "admin":
        return "admin"

    for project in PROJECTS.values():
        if project.get("owner_id") == user_id:
            return "owner"

    return "user"


def can_modify_project(current_user_id: int, project_id: int) -> bool:
    """
    Determine whether the current user can modify the specified project.

    Rules:
    - Admins can modify any project.
    - A user can modify a project if they are the owner of that specific project.
    - All others cannot modify the project.

    Args:
        current_user_id: The ID of the acting user.
        project_id: The ID of the project to check.

    Returns:
        True if the user can modify the project, False otherwise.
    """
    project = PROJECTS.get(project_id)
    if project is None:
        return False

    role = check_user_role(current_user_id)
    if role == "admin":
        return True

    if role == "owner" and project.get("owner_id") == current_user_id:
        return True

    return False
