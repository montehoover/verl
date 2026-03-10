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


def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    """
    Allow project owners to add/remove members in their own projects,
    and admins to manage any project without restriction.

    Returns True if the operation was performed, False otherwise.
    """
    # Validate inputs and existence
    if action_type not in ("add", "remove"):
        return False

    project = PROJECTS.get(prj_id)
    if project is None:
        return False

    active_user = USERS.get(active_user_id)
    target_user = USERS.get(member_user_id)
    if active_user is None or target_user is None:
        return False

    # Permission check
    is_admin = active_user.get("role") == "admin"
    is_owner = project.get("owner_id") == active_user_id
    if not (is_admin or is_owner):
        return False

    # Ensure members list exists and is a list
    members = project.setdefault("members", [])
    if not isinstance(members, list):
        return False

    if action_type == "add":
        if member_user_id in members:
            return False
        members.append(member_user_id)
        return True

    # action_type == "remove"
    if member_user_id not in members:
        return False
    members.remove(member_user_id)
    return True
