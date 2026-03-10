def is_user_admin(user_id: int) -> int:
    """
    Count how many projects the user with the given user_id is a member of.

    Expects a global data structure PROJECTS (a mapping/dict) where each value is a
    project record. Each project record should contain:
      - 'members': an iterable of member identifiers (typically user_id integers).
        Optionally, members can be dicts containing 'user_id' or 'id'.
      - optionally 'active': if present and falsy, the project is considered inactive
        and is not counted.

    Args:
        user_id: The user's integer identifier.

    Returns:
        The number of active projects in PROJECTS where the user appears in the 'members' list.
        Returns 0 if PROJECTS is not defined, not a mapping, or the user is not in any project.
    """
    if not isinstance(user_id, int):
        return 0

    # Access the global projects structure if available.
    try:
        projects = PROJECTS  # type: ignore[name-defined]
    except NameError:
        return 0

    # Ensure we can iterate over project records.
    try:
        project_values = projects.values()  # type: ignore[attr-defined]
    except AttributeError:
        return 0

    count = 0
    for proj in project_values:
        # Skip non-dict project entries
        if not isinstance(proj, dict):
            continue

        # If an 'active' flag exists and is falsy, skip this project.
        if "active" in proj and not proj.get("active"):
            continue

        members = proj.get("members")
        if members is None:
            continue

        # Ensure members is iterable
        try:
            iterator = iter(members)
        except TypeError:
            continue

        found = False
        for m in iterator:
            if m == user_id:
                found = True
                break
            if isinstance(m, dict):
                if m.get("user_id") == user_id or m.get("id") == user_id:
                    found = True
                    break

        if found:
            count += 1

    return count


def manage_project_access(current_user_id: int, project_id: int, action: str, target_user_id: int) -> bool:
    """
    Allow project owners to add/remove members for their project; admins can modify any project.

    Args:
        current_user_id: The user performing the action.
        project_id: The project to modify.
        action: 'add' or 'remove'.
        target_user_id: The user to add or remove.

    Returns:
        True if the action was performed (i.e., membership changed), otherwise False.
    """
    # Basic type validation
    if not isinstance(current_user_id, int) or not isinstance(project_id, int) or not isinstance(target_user_id, int):
        return False
    if not isinstance(action, str):
        return False

    action_norm = action.strip().lower()
    if action_norm not in ("add", "remove"):
        return False

    # Access globals
    try:
        users = USERS  # type: ignore[name-defined]
        projects = PROJECTS  # type: ignore[name-defined]
    except NameError:
        return False

    # Validate project
    proj = projects.get(project_id)
    if not isinstance(proj, dict):
        return False

    # Validate current user and authorization
    cur_user = users.get(current_user_id)
    if not isinstance(cur_user, dict):
        return False

    is_admin = str(cur_user.get("role", "")).lower() == "admin"
    owner_id = proj.get("owner_id")

    if not (is_admin or owner_id == current_user_id):
        return False

    # Validate target user exists
    if target_user_id not in users:
        return False

    # Normalize members list
    members = proj.get("members")
    if members is None:
        proj["members"] = []
        members = proj["members"]
    elif not isinstance(members, list):
        try:
            members = list(members)
        except TypeError:
            members = []
        proj["members"] = members

    # Perform action
    if action_norm == "add":
        if target_user_id in members:
            return False
        members.append(target_user_id)
        return True

    # action_norm == "remove"
    if target_user_id not in members:
        return False
    members.remove(target_user_id)
    return True
