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


def control_project_permissions(executing_user_id: int, prjct_id: int, act_type: str, tgt_user_id: int) -> bool:
    project = PROJECTS.get(prjct_id)
    if project is None:
        return False

    exec_user = USERS.get(executing_user_id)
    tgt_user = USERS.get(tgt_user_id)
    if exec_user is None or tgt_user is None:
        return False

    if act_type not in ("add", "remove"):
        return False

    is_admin = exec_user.get("role") == "admin"
    is_owner = project.get("owner_id") == executing_user_id
    if not (is_admin or is_owner):
        return False

    members = project.get("members")
    if not isinstance(members, list):
        return False

    if act_type == "add":
        if tgt_user_id in members:
            return False
        members.append(tgt_user_id)
        return True

    # act_type == "remove"
    if tgt_user_id not in members:
        return False
    members.remove(tgt_user_id)
    return True
