def is_member_of_project(user_id: int, project_id: int) -> bool:
    """
    Determine whether a user is a member of a given project.

    Expects a global data structure that tracks memberships, e.g.:
    - PROJECT_MEMBERSHIPS: dict[project_id, Iterable[user_id]] or dict[user_id, Iterable[project_id]]
    Optionally:
    - USERS: dict|set|list containing user_id keys/entries.

    Returns True if membership is found, else False.
    """
    # Normalize and validate arguments as integers
    try:
        uid = int(user_id)
        pid = int(project_id)
    except (TypeError, ValueError):
        return False

    # Validate user existence if a USERS collection is available
    users = globals().get("USERS")
    if users is not None:
        if isinstance(users, dict):
            if uid not in users:
                return False
        elif isinstance(users, (set, list, tuple)):
            if uid not in users:
                return False

    # Try known global membership containers
    memberships = (
        globals().get("PROJECT_MEMBERSHIPS")
        or globals().get("PROJECTS")
        or globals().get("PROJECT_MEMBERS")
        or globals().get("PROJECT_MEMBERSHIP")
    )

    if memberships is None:
        return False

    # Case 1: dict-based memberships
    if isinstance(memberships, dict):
        # Orientation A: project_id -> members
        if pid in memberships:
            members = memberships[pid]
            if isinstance(members, dict):
                return uid in members
            if isinstance(members, (set, list, tuple)):
                return uid in members
            try:
                return uid in members  # fallback for other containers
            except TypeError:
                pass

        # Orientation B: user_id -> projects
        if uid in memberships:
            projects = memberships[uid]
            if isinstance(projects, dict):
                return pid in projects
            if isinstance(projects, (set, list, tuple)):
                return pid in projects
            try:
                return pid in projects
            except TypeError:
                pass

        # Orientation C: nested mapping e.g. {project_id: {"members": {...}}}
        container = memberships.get(pid)
        if isinstance(container, dict):
            for key in ("members", "users", "member_ids", "user_ids"):
                if key in container:
                    coll = container[key]
                    if isinstance(coll, dict):
                        return uid in coll
                    if isinstance(coll, (set, list, tuple)):
                        return uid in coll
                    try:
                        return uid in coll
                    except TypeError:
                        pass
        return False

    # Case 2: iterable of membership relations, e.g., set/list of pairs or dicts
    if isinstance(memberships, (set, list, tuple)):
        for item in memberships:
            # Tuple pair: either (project_id, user_id) or (user_id, project_id)
            if isinstance(item, tuple) and len(item) == 2:
                a, b = item
                if (a == pid and b == uid) or (a == uid and b == pid):
                    return True
            # Dict item: {"project_id": ..., "user_id": ...}
            elif isinstance(item, dict):
                item_pid = (
                    item.get("project_id", item.get("project", item.get("pid")))
                )
                item_uid = item.get("user_id", item.get("user", item.get("uid")))
                if item_pid == pid and item_uid == uid:
                    return True
        return False

    # Unknown structure
    return False


def add_user_to_project(user_id: int, project_id: int) -> bool:
    """
    Add a user to a project's members list in the global PROJECTS dict.

    Behavior:
    - Returns True if, after the call, the user is a member of the project.
    - Returns False if PROJECTS or the specified project is missing/invalid,
      or if inputs are invalid.

    PROJECTS is expected to be a dict like:
        { project_id: { "members": [user_id, ...], ... }, ... }

    'members' may be a list, set, tuple, or dict (keys as user IDs).
    """
    try:
        uid = int(user_id)
        pid = int(project_id)
    except (TypeError, ValueError):
        return False

    projects = globals().get("PROJECTS")
    if not isinstance(projects, dict):
        return False

    project = projects.get(pid)
    if not isinstance(project, dict):
        return False

    members = project.get("members")

    # Initialize members if absent
    if members is None:
        project["members"] = [uid]
        return True

    # Normalize types and add membership
    if isinstance(members, dict):
        members[uid] = True
        return True

    if isinstance(members, set):
        members.add(uid)
        return True

    if isinstance(members, list):
        if uid not in members:
            members.append(uid)
        return True

    if isinstance(members, tuple):
        # tuples are immutable; convert to list for mutability
        new_members = list(members)
        if uid not in new_members:
            new_members.append(uid)
        project["members"] = new_members
        return True

    # Fallback: try containment and rebuild as list
    try:
        contains = uid in members  # type: ignore[operator]
    except TypeError:
        contains = False
    if not contains:
        project["members"] = [uid]
    return True


def remove_user_from_project(user_id: int, project_id: int) -> bool:
    """
    Remove a user from a project's members list in the global PROJECTS dict.

    Behavior:
    - Returns True if, after the call, the user is not a member of the project.
    - Returns False if PROJECTS or the specified project is missing/invalid,
      or if inputs are invalid.

    PROJECTS is expected to be a dict like:
        { project_id: { "members": [user_id, ...], ... }, ... }
    """
    try:
        uid = int(user_id)
        pid = int(project_id)
    except (TypeError, ValueError):
        return False

    projects = globals().get("PROJECTS")
    if not isinstance(projects, dict):
        return False

    project = projects.get(pid)
    if not isinstance(project, dict):
        return False

    members = project.get("members")

    if members is None:
        return True  # already not a member

    if isinstance(members, dict):
        members.pop(uid, None)
        return True

    if isinstance(members, set):
        members.discard(uid)
        return True

    if isinstance(members, list):
        try:
            members.remove(uid)
        except ValueError:
            pass
        return True

    if isinstance(members, tuple):
        # Convert to list and remove if present
        new_members = [m for m in members if m != uid]
        project["members"] = new_members
        return True

    # Unknown members type; ensure user not present by resetting to filtered list if possible
    try:
        contains = uid in members  # type: ignore[operator]
    except TypeError:
        contains = False
    if contains:
        project["members"] = [m for m in members if m != uid]  # type: ignore[union-attr]
    return True


def is_user_admin(user_id: int) -> int:
    """
    Count how many projects in the global PROJECTS dictionary include user_id
    in their 'members' collection. Returns 0 if none or on invalid input.
    """
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return 0

    projects = globals().get("PROJECTS")
    if not isinstance(projects, dict):
        return 0

    # Optional user existence check
    users = globals().get("USERS")
    if users is not None:
        if isinstance(users, dict) and uid not in users:
            return 0
        if isinstance(users, (set, list, tuple)) and uid not in users:
            return 0

    count = 0
    for proj in projects.values():
        if not isinstance(proj, dict):
            continue
        members = proj.get("members")
        if members is None:
            continue

        if isinstance(members, dict):
            if uid in members:
                count += 1
        elif isinstance(members, (set, list, tuple)):
            if uid in members:
                count += 1
        else:
            try:
                if uid in members:
                    count += 1
            except TypeError:
                continue

    return count
