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


def has_project_management_permission(active_user: dict, project: dict) -> bool:
    """
    Pure function: Determine if the active_user can manage the given project.
    Admins can manage any project; owners can manage their own project.
    """
    if not active_user or not project:
        return False
    return active_user.get("role") == "admin" or project.get("owner_id") == active_user.get("user_id")


def compute_members_update(action_type: str, members, member_user_id: int):
    """
    Pure function: Compute the new members list based on the requested action.
    Returns (success: bool, new_members: list). Does not mutate the input list.
    """
    if not isinstance(members, list):
        return False, members

    if action_type == "add":
        if member_user_id in members:
            return False, members
        return True, members + [member_user_id]

    # action_type == "remove"
    if member_user_id not in members:
        return False, members
    new_members = [m for m in members if m != member_user_id]
    return True, new_members


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

    # Permission check (pure)
    if not has_project_management_permission(active_user, project):
        return False

    # Ensure members key exists (maintains previous behavior)
    members = project.setdefault("members", [])

    # Compute new members list (pure)
    success, new_members = compute_members_update(action_type, members, member_user_id)
    if not success:
        return False

    # Apply update to state
    project["members"] = new_members
    return True
