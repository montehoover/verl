def is_user_admin(user_id: int) -> int:
    """
    Count how many active projects a user is a member of.

    Expects a global dictionary named PROJECTS where each value is a project dict.
    A project is considered active if:
      - it has an 'active' key that is truthy; or
      - it has no 'active' key (treated as active by default).

    A user is considered a member if their user_id appears in the project's 'members'
    iterable (commonly a list). Members may be:
      - integers (user IDs)
      - strings convertible to integers
      - dicts containing 'user_id' or 'id' equal to the user_id
      - a mapping (e.g., dict) where keys are user IDs
    """
    projects = globals().get("PROJECTS")
    if projects is None:
        return 0

    # Determine iterable of project entries
    try:
        iterable = projects.values() if hasattr(projects, "values") else iter(projects)
    except Exception:
        return 0

    count = 0
    for proj in iterable:
        if not isinstance(proj, dict):
            # Skip malformed project entries
            continue

        # Active check: if 'active' key exists, it must be truthy; otherwise treat as active.
        if "active" in proj and not proj.get("active"):
            continue

        members = proj.get("members")
        if members is None:
            continue

        try:
            # If members is a mapping, consider its keys as member identifiers.
            if hasattr(members, "keys"):
                member_iter = members.keys()
            else:
                member_iter = members

            is_member = False
            for m in member_iter:
                if isinstance(m, int):
                    if m == user_id:
                        is_member = True
                        break
                elif isinstance(m, dict):
                    mid = m.get("user_id", m.get("id"))
                    if isinstance(mid, int) and mid == user_id:
                        is_member = True
                        break
                    try:
                        if int(mid) == user_id:
                            is_member = True
                            break
                    except Exception:
                        pass
                else:
                    try:
                        if int(m) == user_id:
                            is_member = True
                            break
                    except Exception:
                        continue

            if is_member:
                count += 1
        except Exception:
            # Skip problematic project entries/members lists
            continue

    return count


def control_project_permissions(executing_user_id: int, prjct_id: int, act_type: str, tgt_user_id: int) -> bool:
    """
    Allow project owners to add/remove members from their projects,
    while admins can manage any project.

    Args:
        executing_user_id: The user performing the action.
        prjct_id: The project to modify.
        act_type: 'add' or 'remove'.
        tgt_user_id: The user to add or remove.

    Returns:
        True if the operation was successfully completed (i.e., a mutation occurred), else False.

    Globals expected:
        USERS: dict[int, dict] with a 'role' key ('admin' or 'user').
        PROJECTS: dict[int, dict] with 'owner_id' and 'members' list.
    """
    users = globals().get("USERS")
    projects = globals().get("PROJECTS")
    if not isinstance(users, dict) or not isinstance(projects, dict):
        return False

    project = projects.get(prjct_id)
    if not isinstance(project, dict):
        return False

    exec_user = users.get(executing_user_id)
    if not isinstance(exec_user, dict):
        return False

    role = str(exec_user.get("role", "")).strip().lower()
    is_admin = role == "admin"
    is_owner = project.get("owner_id") == executing_user_id

    if not (is_admin or is_owner):
        return False

    target_user = users.get(tgt_user_id)
    if not isinstance(target_user, dict):
        return False

    action = str(act_type).strip().lower()
    if action not in ("add", "remove"):
        return False

    members = project.get("members")
    # Normalize members to a mutable list
    if members is None:
        members_list = []
    elif isinstance(members, list):
        members_list = members
    elif isinstance(members, set):
        members_list = list(members)
    elif isinstance(members, tuple):
        members_list = list(members)
    elif hasattr(members, "keys"):
        # If a mapping is used, take its keys as members
        try:
            members_list = list(members.keys())
        except Exception:
            return False
    else:
        try:
            members_list = list(members)
        except Exception:
            return False

    def member_index(lst, uid: int):
        for i, m in enumerate(lst):
            if isinstance(m, int):
                if m == uid:
                    return i
            elif isinstance(m, dict):
                mid = m.get("user_id", m.get("id"))
                try:
                    if int(mid) == uid:
                        return i
                except Exception:
                    continue
            else:
                try:
                    if int(m) == uid:
                        return i
                except Exception:
                    continue
        return None

    idx = member_index(members_list, tgt_user_id)

    if action == "add":
        if idx is not None:
            # No mutation
            return False
        members_list.append(tgt_user_id)
        project["members"] = members_list
        return True

    # action == "remove"
    if idx is None:
        # No mutation
        return False
    members_list.pop(idx)
    project["members"] = members_list
    return True
