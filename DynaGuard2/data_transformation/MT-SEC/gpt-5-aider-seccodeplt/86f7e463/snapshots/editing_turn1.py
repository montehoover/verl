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
