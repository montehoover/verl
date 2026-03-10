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


ActionStr = Literal["add", "remove"]


def manage_project_access(
    current_user_id: int, project_id: int, action: ActionStr, target_user_id: int
) -> bool:
    """
    Allow project owners to add/remove members of their projects, and admins to modify any project.

    Rules:
    - Only admins or the owner of the specific project can modify membership.
    - action must be "add" or "remove".
    - target_user_id must exist in USERS.
    - Adding is idempotent (adding an existing member returns True).
    - Removing is idempotent (removing a non-member returns True).
    - The project owner cannot be removed from their project.

    Returns:
        True if the action succeeds (or results in no change due to idempotency), else False.
    """
    project = PROJECTS.get(project_id)
    if project is None:
        return False

    if target_user_id not in USERS:
        return False

    if action not in ("add", "remove"):
        return False

    if not can_modify_project(current_user_id, project_id):
        return False

    members: List[int] = project.get("members", [])
    owner_id = project.get("owner_id")

    if action == "add":
        if target_user_id in members:
            return True  # idempotent success
        members.append(target_user_id)
        project["members"] = members
        return True

    # action == "remove"
    if target_user_id == owner_id:
        return False  # cannot remove owner

    if target_user_id not in members:
        return True  # idempotent success

    project["members"] = [m for m in members if m != target_user_id]
    return True
