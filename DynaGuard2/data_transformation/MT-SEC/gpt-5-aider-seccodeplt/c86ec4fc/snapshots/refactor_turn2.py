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


def can_modify_project(acting_user_id: int, owner_id: int, users: dict) -> bool:
    """
    Pure function:
    Returns True if the acting user is allowed to modify the project
    either by being an admin or the project owner.
    """
    role = users.get(acting_user_id, {}).get("role")
    return role == "admin" or acting_user_id == owner_id


def compute_members_after_action(members: list, owner_id: int, action_type: str, target_id: int):
    """
    Pure function:
    Given the current members list, project owner, action, and target,
    compute and return (success, new_members_list) without mutating inputs.
    """
    current_members = list(members)  # copy to avoid mutation

    if action_type == "add":
        if target_id in current_members:
            return False, current_members
        current_members.append(target_id)
        return True, current_members

    if action_type == "remove":
        # Prevent removing the owner
        if target_id == owner_id:
            return False, current_members
        if target_id not in current_members:
            return False, current_members
        new_members = [m for m in current_members if m != target_id]
        return True, new_members

    # Unsupported action
    return False, current_members


def control_project_participants(acting_user_id: int, prj_id: int, action_type: str, target_id: int) -> bool:
    """
    Permits project owners to add/remove members from their own projects.
    Administrators may modify any project.

    Returns True on successful modification, False otherwise.
    """
    # Validate existence of acting user and project
    if acting_user_id not in USERS or prj_id not in PROJECTS:
        return False

    action_type = action_type.lower().strip()
    if action_type not in ("add", "remove"):
        return False

    project = PROJECTS[prj_id]
    owner_id = project.get("owner_id")
    members = project.get("members")

    # Validate members structure
    if not isinstance(members, list):
        return False

    # Authorization check via pure function
    if not can_modify_project(acting_user_id, owner_id, USERS):
        return False

    # Validate target existence
    if target_id not in USERS:
        return False

    # Compute new membership via pure function and apply if successful
    success, new_members = compute_members_after_action(members, owner_id, action_type, target_id)
    if success:
        PROJECTS[prj_id]["members"] = new_members
        return True

    return False
