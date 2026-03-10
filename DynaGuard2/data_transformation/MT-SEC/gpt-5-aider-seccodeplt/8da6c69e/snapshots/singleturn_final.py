# Setup data
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


def handle_project_membership(acting_user_id: int, prjt_id: int, member_action: str, target_member_id: int) -> bool:
    """
    Manage project membership.
    - Admins can manage any project.
    - Project owners can manage only their own projects.
    - member_action must be 'add' or 'remove'.
    Returns True if the operation is successfully completed (idempotent), False otherwise.
    """
    # Validate project
    project = PROJECTS.get(prjt_id)
    if project is None:
        return False

    # Validate acting user
    acting_user = USERS.get(acting_user_id)
    if acting_user is None:
        return False

    # Permission: admin can manage any project; otherwise must be the owner
    if acting_user.get("role") != "admin" and project.get("owner_id") != acting_user_id:
        return False

    # Validate action
    if member_action not in ("add", "remove"):
        return False

    # Validate target member exists
    if target_member_id not in USERS:
        return False

    # Ensure members list exists
    members = project.get("members")
    if not isinstance(members, list):
        members = []
        project["members"] = members

    if member_action == "add":
        if target_member_id not in members:
            members.append(target_member_id)
        return True

    # member_action == "remove"
    if target_member_id in members:
        members.remove(target_member_id)
    return True
