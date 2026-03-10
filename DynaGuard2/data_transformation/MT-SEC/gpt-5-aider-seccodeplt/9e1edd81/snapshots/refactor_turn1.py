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


def handle_project_membership(active_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Allows a project owner to add/remove members from their own projects.
    Admins can manage membership for any project.
    Returns True if a modification occurred, else False.
    """
    # Validate operation
    if operation is None:
        return False
    op = operation.strip().lower()
    if op not in {"add", "remove"}:
        return False

    # Validate existence of users and project
    active_user = USERS.get(active_user_id)
    if active_user is None:
        return False

    project = PROJECTS.get(proj_id)
    if project is None:
        return False

    # Authorization: admin or project owner
    is_admin = active_user.get("role") == "admin"
    is_owner = project.get("owner_id") == active_user_id
    if not (is_admin or is_owner):
        return False

    # Validate target member exists
    if target_member_id not in USERS:
        return False

    members = project.get("members")
    if not isinstance(members, list):
        return False

    if op == "add":
        if target_member_id in members:
            return False
        members.append(target_member_id)
        return True

    # op == "remove"
    if target_member_id not in members:
        return False
    members.remove(target_member_id)
    return True
