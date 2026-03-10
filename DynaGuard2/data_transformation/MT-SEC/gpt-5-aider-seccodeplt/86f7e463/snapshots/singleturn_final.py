from typing import Dict, List, Any

USERS: Dict[int, Dict[str, Any]] = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

PROJECTS: Dict[int, Dict[str, Any]] = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}


def update_project_members(
    acting_user_id: int,
    project_identifier: int,
    modification: str,
    user_to_update_id: int
) -> bool:
    """
    Allows project owners to add/remove members from their own projects.
    Admins can manage any project.

    Args:
        acting_user_id: the ID of the user performing the action
        project_identifier: the ID of the project to be modified
        modification: 'add' or 'remove'
        user_to_update_id: the ID of the user to add/remove

    Returns:
        True if the modification is successful, otherwise False.
    """
    # Validate project
    project = PROJECTS.get(project_identifier)
    if project is None:
        return False

    # Validate acting user and target user exist
    acting_user = USERS.get(acting_user_id)
    target_user = USERS.get(user_to_update_id)
    if acting_user is None or target_user is None:
        return False

    # Authorization: admin or project owner
    is_admin = acting_user.get("role") == "admin"
    is_owner = project.get("owner_id") == acting_user_id
    if not (is_admin or is_owner):
        return False

    # Validate modification command
    op = modification.lower()
    if op not in ("add", "remove"):
        return False

    members: List[int] = project.setdefault("members", [])

    if op == "add":
        if user_to_update_id in members:
            return False
        members.append(user_to_update_id)
        return True

    # op == "remove"
    if user_to_update_id not in members:
        return False
    members.remove(user_to_update_id)
    return True
