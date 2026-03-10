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


def manage_team_access(executor_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Manage project team access:
    - Admins can add/remove members on any project.
    - Project owners can add/remove members on their own projects only.

    Returns True if the requested operation successfully changed membership, otherwise False.
    """
    # Validate inputs
    if operation not in ("add", "remove"):
        return False

    # Check existence of executor, project, and target user
    executor = USERS.get(executor_user_id)
    if executor is None:
        return False

    project = PROJECTS.get(proj_id)
    if project is None:
        return False

    if target_member_id not in USERS:
        return False

    # Permission check: admin or project owner
    is_admin = executor.get("role") == "admin"
    is_owner = project.get("owner_id") == executor_user_id
    if not (is_admin or is_owner):
        return False

    members = project.get("members")
    if not isinstance(members, list):
        return False

    # Prevent removing the project owner from members
    if operation == "remove" and target_member_id == project.get("owner_id"):
        return False

    # Execute operation
    if operation == "add":
        if target_member_id in members:
            return False
        members.append(target_member_id)
        return True

    if operation == "remove":
        if target_member_id not in members:
            return False
        members.remove(target_member_id)
        return True

    return False
