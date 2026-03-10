def handle_project_membership(
    acting_user_id: int,
    prjt_id: int,
    member_action: str,
    target_member_id: int
) -> bool:
    """
    Manage project membership based on permissions.

    - Admins can add/remove members from any project.
    - Project owners can add/remove members from their own projects.
    - Returns True if the operation completes successfully (including no-op if state already satisfied), False otherwise.
    """
    # Validate existence of global data stores
    try:
        users = USERS  # type: ignore[name-defined]
        projects = PROJECTS  # type: ignore[name-defined]
    except NameError:
        return False

    # Validate project and acting user existence
    project = projects.get(prjt_id)
    actor = users.get(acting_user_id)
    if project is None or actor is None:
        return False

    # Normalize action
    if not isinstance(member_action, str):
        return False
    action = member_action.strip().lower()
    if action not in ("add", "remove"):
        return False

    # Permission checks
    is_admin = actor.get("role") == "admin"
    is_owner = acting_user_id == project.get("owner_id")
    if not (is_admin or is_owner):
        return False

    # Fetch members list
    members = project.get("members")
    if not isinstance(members, list):
        return False

    # Execute action
    if action == "add":
        # Target must be a known user to be added
        if target_member_id not in users:
            return False
        if target_member_id in members:
            return True  # already a member; idempotent success
        members.append(target_member_id)
        return True

    # action == "remove"
    # Do not allow removing the project owner from membership
    if target_member_id == project.get("owner_id"):
        return False
    if target_member_id not in members:
        return True  # already not a member; idempotent success
    # Remove all occurrences just in case
    project["members"] = [uid for uid in members if uid != target_member_id]
    return True
