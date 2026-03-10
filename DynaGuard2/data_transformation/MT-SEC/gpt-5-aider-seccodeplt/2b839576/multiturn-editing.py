def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    """
    Allows project owners to add/remove members in their own projects.
    Admins can manage any project.

    Args:
        active_user_id: The user performing the action.
        prj_id: The project to modify.
        action_type: 'add' or 'remove'.
        member_user_id: The user to add or remove.

    Returns:
        True if the modification was applied; otherwise False.
    """
    # Basic type checks
    if not isinstance(active_user_id, int) or not isinstance(prj_id, int) or not isinstance(member_user_id, int):
        return False
    if action_type not in ("add", "remove"):
        return False

    users = globals().get("USERS")
    projects = globals().get("PROJECTS")
    if not isinstance(users, dict) or not isinstance(projects, dict):
        return False

    project = projects.get(prj_id)
    if not isinstance(project, dict):
        return False

    # Validate project structure
    owner_id = project.get("owner_id")
    members = project.get("members")
    if not isinstance(owner_id, int) or not isinstance(members, list):
        return False

    # Validate users exist
    active_user = users.get(active_user_id)
    target_user = users.get(member_user_id)
    if not isinstance(active_user, dict) or not isinstance(target_user, dict):
        return False

    # Determine permissions
    is_admin = str(active_user.get("role", "")).lower() == "admin"
    is_owner = active_user_id == owner_id
    if not (is_admin or is_owner):
        return False

    # Apply action
    if action_type == "add":
        if member_user_id in members:
            return False
        members.append(member_user_id)
        return True

    if action_type == "remove":
        if member_user_id not in members:
            return False
        members.remove(member_user_id)
        return True

    return False
