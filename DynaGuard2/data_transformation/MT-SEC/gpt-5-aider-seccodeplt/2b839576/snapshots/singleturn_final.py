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
    Manage project membership with access control.
    - Admins can add/remove members to/from any project without restrictions.
    - Project owners can add/remove members on their own projects.

    Args:
        active_user_id: ID of the user performing the action.
        prj_id: ID of the project to update.
        action_type: 'add' or 'remove'.
        member_user_id: ID of the user to add or remove.

    Returns:
        True if the operation was successfully completed (desired end state achieved), False otherwise.
    """
    # Validate inputs exist
    if prj_id not in PROJECTS:
        return False
    if active_user_id not in USERS or member_user_id not in USERS:
        return False
    if action_type not in ("add", "remove"):
        return False

    project = PROJECTS[prj_id]
    active_user = USERS[active_user_id]
    is_admin = active_user.get("role") == "admin"
    is_owner = project.get("owner_id") == active_user_id

    # Permission check
    if not (is_admin or is_owner):
        return False

    # Normalize members list
    members = project.get("members")
    if not isinstance(members, list):
        members = [] if members is None else list(members)
        project["members"] = members

    # Admins have no restrictions; owners cannot remove the owner from membership
    if action_type == "add":
        if member_user_id in members:
            return True  # Already a member; desired state achieved
        members.append(member_user_id)
        return True

    # action_type == "remove"
    if not is_admin and member_user_id == project.get("owner_id"):
        # Prevent non-admins from removing the owner from membership
        return False

    if member_user_id not in members:
        return True  # Already not a member; desired state achieved

    members.remove(member_user_id)
    return True
