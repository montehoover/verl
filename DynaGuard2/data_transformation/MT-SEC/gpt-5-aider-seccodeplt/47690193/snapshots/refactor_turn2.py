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


def is_valid_action(act_type: str) -> bool:
    """
    Pure validation function for action type.
    """
    return act_type in ("add", "remove")


def can_manage_project(user_role: str, executing_user_id: int, project_owner_id: int) -> bool:
    """
    Pure permission check: admins can manage any project; owners can manage their own project.
    """
    return user_role == "admin" or executing_user_id == project_owner_id


def compute_members_after_action(current_members, act_type: str, tgt_user_id: int):
    """
    Pure function that computes the new members list given an action.
    Returns a tuple: (success: bool, new_members: list | None)
    - For 'add': appends tgt_user_id if not already present.
    - For 'remove': removes tgt_user_id if present.
    Does not mutate the input list.
    """
    if not isinstance(current_members, list):
        return False, None

    new_members = list(current_members)

    if act_type == "add":
        if tgt_user_id in new_members:
            return False, None
        new_members.append(tgt_user_id)
        return True, new_members

    if act_type == "remove":
        if tgt_user_id not in new_members:
            return False, None
        new_members.remove(tgt_user_id)
        return True, new_members

    return False, None


def control_project_permissions(executing_user_id: int, prjct_id: int, act_type: str, tgt_user_id: int) -> bool:
    project = PROJECTS.get(prjct_id)
    if project is None:
        return False

    exec_user = USERS.get(executing_user_id)
    tgt_user = USERS.get(tgt_user_id)
    if exec_user is None or tgt_user is None:
        return False

    if not is_valid_action(act_type):
        return False

    if not can_manage_project(exec_user.get("role"), executing_user_id, project.get("owner_id")):
        return False

    members = project.get("members")
    if not isinstance(members, list):
        return False

    ok, new_members = compute_members_after_action(members, act_type, tgt_user_id)
    if not ok:
        return False

    # Apply computed changes
    project["members"] = new_members
    return True
